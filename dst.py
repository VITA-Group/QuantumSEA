from __future__ import print_function

import pdb
import math
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class CosineDecay(object):
    def __init__(self, death_rate, T_max, eta_min=0.005, last_epoch=-1):
        self.sgd = optim.SGD(torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1))]), lr=death_rate)
        self.cosine_stepper = torch.optim.lr_scheduler.CosineAnnealingLR(self.sgd, T_max, eta_min, last_epoch)

    def step(self):
        self.cosine_stepper.step()

    def get_dr(self, death_rate):
        return self.sgd.param_groups[0]['lr']

class LinearDecay(object):
    def __init__(self, death_rate, factor=0.99, frequency=600):
        self.factor = factor
        self.steps = 0
        self.frequency = frequency

    def step(self):
        self.steps += 1

    def get_dr(self, death_rate):
        if self.steps > 0 and self.steps % self.frequency == 0:
            return death_rate*self.factor
        else:
            return death_rate

class Masking(object):
    def __init__(self, optimizer, 
        death_rate=0.1,
        death_rate_decay=None, 
        death_mode='magnitude', 
        growth_mode='gradient_prob',
        args=None,
        decay_schedule='cosine'):

        self.args = args
        self.device = torch.device("cuda")
        self.optimizer = optimizer
        self.steps = 0
        self.optim = optim

        self.masks = {}
        self.names = []
        self.model = None     
        self.masks_shape = {}
        self.gradient_accumulator = {} # Store accumulated gradient
        self.ops_sample = []
        self.trunc_masks = {}

        self.death_mode = death_mode
        self.death_rate = death_rate
        self.decay_schedule = decay_schedule
        self.death_rate_decay = death_rate_decay

        self.growth_mode = growth_mode
        self.growth_rate = death_rate

        self.init_mode = None


    def add_module(self, module, density, reset_operation=False, sparse_init='uniform_global'):
        self.model = module
        for name, tensor in self.model.named_parameters():
            name_mask = name + '_mask'
            self.names.append(name_mask)
            self.masks[name_mask] = torch.zeros_like(tensor, dtype=torch.float32, requires_grad=False).cuda()
            self.trunc_masks[name_mask] = torch.zeros_like(tensor, dtype=torch.float32, requires_grad=False).cuda()
            self.gradient_accumulator[name] = torch.zeros_like(tensor, dtype=torch.float32, requires_grad=False).cuda()
            self.masks_shape[name_mask] = tensor.shape
        self.init(mode=sparse_init, density=density)
        if reset_operation:
            self.reset_operation_gate()

    def step(self, beta=0.9):
        self.apply_mask2grad()
        self.optimizer.step()
        self.gradient_magnitude_accumulate(beta)
        self.death_rate_decay.step()
        if self.decay_schedule == 'cosine':
            self.death_rate = self.death_rate_decay.get_dr(self.death_rate)
            self.growth_rate = self.death_rate
        self.steps += 1

    def update_structure(self, beta_random=1.0, reset_operation=False):
        before_masks = copy.deepcopy(self.masks)

        # death
        if self.death_mode == 'magnitude':
            self.magnitude_death()
        elif self.death_mode == 'gradient':
            self.gradient_death()
        elif self.death_mode == 'saliency':
            self.saliency_death()
        elif self.death_mode == 'random':
            self.random_death()
        
        # growth
        if self.growth_mode == 'gradient_prob':
            self.gradient_probabilistic_growth(beta_random)
        elif self.growth_mode == 'random':
            self.gradient_probabilistic_growth(1.0)
        elif self.growth_mode == 'gradient':
            self.gradient_probabilistic_growth(0)
        
        self.apply_mask2weight()

        # show update result
        for key in before_masks:
            print('key = {}, from {} to {}'.format(key, before_masks[key], self.masks[key]))
        self.show_density()
        if reset_operation:
            self.reset_operation_gate()


    # basic function
    def init(self, mode='uniform_global', density=0.50):
        self.density = density
        self.init_mode = mode

        if mode == 'uniform_global':
            sum_params = 0
            for key in self.masks_shape:
                sum_params += np.prod(self.masks_shape[key])
            remain_params = int(density * sum_params)
            remain_tensor = torch.zeros(sum_params)
            pruned_index = np.random.permutation(sum_params)[:remain_params]
            remain_tensor[pruned_index] = 1
            cnt_current = 0
            for key in self.masks:
                num = np.prod(self.masks[key].shape)
                self.masks[key] = remain_tensor[cnt_current:num+cnt_current].reshape(self.masks_shape[key]).cuda()
                cnt_current += num

        elif mode == 'dense_4_50':
            sum_params = 0
            for key in self.masks_shape:
                sum_params += np.prod(self.masks_shape[key])
            remain_tensor = torch.tensor([1,1,0,0,0,0,1,1] * 100)
            remain_tensor = remain_tensor[:sum_params]
            cnt_current = 0
            for key in self.masks:
                num = np.prod(self.masks[key].shape)
                self.masks[key] = remain_tensor[cnt_current:num+cnt_current].reshape(self.masks_shape[key]).cuda()
                cnt_current += num

        elif mode == 'dense_gate_50':

            cnt_current = 0
            design = [1,1,0,0,0,0,1,1]
            for key in self.masks:
                index = cnt_current%len(design)
                self.masks[key] = design[index] * torch.ones_like(self.masks[key])
                cnt_current += 1

        elif mode == 'dense_gate_20':

            cnt_current = 0
            design = [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]
            for key in self.masks:
                index = cnt_current%len(design)
                self.masks[key] = design[index] * torch.ones_like(self.masks[key])
                cnt_current += 1

        elif mode == 'dense_gate_50_2':
            cnt_current = 0
            design = [1,0,0,1]
            for key in self.masks:
                index = cnt_current%len(design)
                self.masks[key] = design[index] * torch.ones_like(self.masks[key])
                cnt_current += 1

        self.apply_mask2weight()
        self.show_density()

    def apply_mask2grad(self):
        for name, weight in self.model.named_parameters():
            name_mask = name + '_mask'
            weight.grad = weight.grad*self.masks[name_mask]

    def apply_mask2weight(self):
        for name, weight in self.model.named_parameters():
            name_mask = name + '_mask' 
            weight.data = weight.data*self.masks[name_mask]
            if 'momentum_buffer' in self.optimizer.state[weight]:
                self.optimizer.state[weight]['momentum_buffer'] = self.optimizer.state[weight]['momentum_buffer']*self.masks[name_mask]
            if 'exp_avg' in self.optimizer.state[weight]:
                self.optimizer.state[weight]['exp_avg'] = self.optimizer.state[weight]['exp_avg']*self.masks[name_mask]
            if 'exp_avg_sq' in self.optimizer.state[weight]:
                self.optimizer.state[weight]['exp_avg_sq'] = self.optimizer.state[weight]['exp_avg_sq']*self.masks[name_mask]
            if 'max_exp_avg_sq' in self.optimizer.state[weight]:
                self.optimizer.state[weight]['max_exp_avg_sq'] = self.optimizer.state[weight]['max_exp_avg_sq']*self.masks[name_mask]

    def gradient_magnitude_accumulate(self, beta=0.9):
        for name, weight in self.model.named_parameters():
            current_grad = weight.grad.data.abs()
            accumulate_grad = self.gradient_accumulator[name]
            self.gradient_accumulator[name] = beta * accumulate_grad + (1 - beta) * current_grad

    def reset_operation_gate(self):

        store_current_weight = {}
        for name, weight in self.model.named_parameters():
            store_current_weight[name] = weight.data.clone()
            name_mask = name + '_mask'
            weight.data = self.masks[name_mask]
        
        self.model.q_layer.reset_sampling()

        # recover weight
        for name, weight in self.model.named_parameters():
            weight.data = store_current_weight[name]

    # main function
    def magnitude_death(self):
        print('Pruning Gate based on Magnitude')
        original_masks = copy.deepcopy(self.masks)

        # collect score
        All_score = []
        new_mask_name = []
        new_mask_shape = []
        for name, weight in self.model.named_parameters():
            name_mask = name + '_mask'
            All_score.append(weight.data.abs().reshape(-1))
            new_mask_name.append(name_mask)
            new_mask_shape.append(weight.data.shape)
        All_score = torch.cat(All_score)

        # remove weight
        remain_num = int(All_score.nelement() * (self.density - self.death_rate))
        remain_index = All_score.sort()[1][-remain_num:]
        new_mask_plain = torch.zeros_like(All_score)
        new_mask_plain[remain_index] = 1
        cnt_current = 0
        for index, mask_name in enumerate(new_mask_name):
            num = np.prod(new_mask_shape[index])
            self.masks[mask_name] = new_mask_plain[cnt_current:num+cnt_current].reshape(new_mask_shape[index]).cuda()
            cnt_current += num

        # check pruning mask 
        illegal_ele = 0
        for key in original_masks:
            illegal_ele += ((1 - original_masks[key]) * self.masks[key]).sum() # illegal: 0 -> 1
        print('Illegal element number = {}'.format(illegal_ele))

    # main function
    def random_death(self):
        print('Pruning Gate based on Random')
        original_masks = copy.deepcopy(self.masks)

        # collect score
        All_score = []
        new_mask_name = []
        new_mask_shape = []
        for key in self.masks:
            cur_mask = self.masks[key]
            cur_score = torch.rand_like(cur_mask) * cur_mask
            All_score.append(cur_score.reshape(-1))
            new_mask_name.append(key)
            new_mask_shape.append(cur_mask.shape)
        All_score = torch.cat(All_score)

        # remove weight
        remain_num = int(All_score.nelement() * (self.density - self.death_rate))
        remain_index = All_score.sort()[1][-remain_num:]
        new_mask_plain = torch.zeros_like(All_score)
        new_mask_plain[remain_index] = 1
        cnt_current = 0
        for index, mask_name in enumerate(new_mask_name):
            num = np.prod(new_mask_shape[index])
            self.masks[mask_name] = new_mask_plain[cnt_current:num+cnt_current].reshape(new_mask_shape[index]).cuda()
            cnt_current += num

        # check pruning mask 
        illegal_ele = 0
        for key in original_masks:
            illegal_ele += ((1 - original_masks[key]) * self.masks[key]).sum() # illegal: 0 -> 1
        print('Illegal element number = {}'.format(illegal_ele))

    def gradient_death(self):
        print('Pruning Gate based on accumulated gradient')
        original_masks = copy.deepcopy(self.masks)

        # collect score
        All_score = []
        new_mask_name = []
        new_mask_shape = []
        for name, weight in self.model.named_parameters():
            name_mask = name + '_mask'
            All_score.append(self.gradient_accumulator[name].reshape(-1))
            new_mask_name.append(name_mask)
            new_mask_shape.append(self.gradient_accumulator[name].shape)
        All_score = torch.cat(All_score)

        # remove weight
        remain_num = int(All_score.nelement() * (self.density - self.death_rate))
        remain_index = All_score.sort()[1][-remain_num:]
        new_mask_plain = torch.zeros_like(All_score)
        new_mask_plain[remain_index] = 1
        cnt_current = 0
        for index, mask_name in enumerate(new_mask_name):
            num = np.prod(new_mask_shape[index])
            self.masks[mask_name] = new_mask_plain[cnt_current:num+cnt_current].reshape(new_mask_shape[index]).cuda()
            cnt_current += num

        # check pruning mask 
        illegal_ele = 0
        for key in original_masks:
            illegal_ele += ((1 - original_masks[key]) * self.masks[key]).sum() # illegal: 0 -> 1
        print('Illegal element number = {}'.format(illegal_ele))

    def saliency_death(self):
        print('Pruning Gate based on Saliency')
        original_masks = copy.deepcopy(self.masks)

        # collect score
        All_score = []
        new_mask_name = []
        new_mask_shape = []
        for name, weight in self.model.named_parameters():
            name_mask = name + '_mask'
            cur_score = weight.grad.data * weight.data
            All_score.append(cur_score.abs().reshape(-1))
            new_mask_name.append(name_mask)
            new_mask_shape.append(cur_score.shape)
        All_score = torch.cat(All_score)

        # remove weight
        remain_num = int(All_score.nelement() * (self.density - self.death_rate))
        remain_index = All_score.sort()[1][-remain_num:]
        new_mask_plain = torch.zeros_like(All_score)
        new_mask_plain[remain_index] = 1
        cnt_current = 0
        for index, mask_name in enumerate(new_mask_name):
            num = np.prod(new_mask_shape[index])
            self.masks[mask_name] = new_mask_plain[cnt_current:num+cnt_current].reshape(new_mask_shape[index]).cuda()
            cnt_current += num

        # check pruning mask 
        illegal_ele = 0
        for key in original_masks:
            illegal_ele += ((1 - original_masks[key]) * self.masks[key]).sum() # illegal: 0 -> 1
        print('Illegal element number = {}'.format(illegal_ele))

    def gradient_probabilistic_growth(self, beta=1.0):
        print('Random Growth based on accumulated gradient, Radnom_beta = {}'.format(beta))
        original_masks = copy.deepcopy(self.masks)

        # collect score
        offset = 1e+20
        gradient_score = []
        new_mask_name = []
        new_mask_shape = []
        offset_score = []
        for name, weight in self.model.named_parameters():
            name_mask = name + '_mask'
            gradient_score.append(self.gradient_accumulator[name].reshape(-1))
            offset_score.append(self.masks[name_mask].reshape(-1))
            new_mask_name.append(name_mask)
            new_mask_shape.append(self.gradient_accumulator[name].shape)
        gradient_score = torch.cat(gradient_score)
        offset_score = torch.cat(offset_score)
        random_score = torch.rand_like(gradient_score)
        # normalize random score 
        mean_gradient = gradient_score.mean().item()
        mean_random = random_score.mean().item()
        random_score = random_score * (mean_gradient / mean_random)
        # mix score 
        All_score = beta * random_score + (1 - beta) * gradient_score  # beta=1: random growth, beta=0: gradient growth
        All_score = All_score + offset_score * offset

        # remove weight
        remain_num = int(All_score.nelement() * (self.density - self.death_rate + self.growth_rate))
        remain_index = All_score.sort()[1][-remain_num:]
        new_mask_plain = torch.zeros_like(All_score)
        new_mask_plain[remain_index] = 1

        cnt_current = 0
        for key in self.masks:
            num = np.prod(self.masks_shape[key])
            self.masks[key] = new_mask_plain[cnt_current:num+cnt_current].reshape(self.masks_shape[key]).cuda()
            cnt_current += num

        # check pruning mask 
        illegal_ele = 0
        for key in original_masks:
            illegal_ele += (original_masks[key] * (1 - self.masks[key])).sum() # illegal: 1->0
        print('Growth Illegal element number = {}'.format(illegal_ele))

    # logging
    def show_density(self):
        total_size = 0
        for name, weight in self.masks.items():
            total_size += weight.numel()
            self.trunc_masks[name] = ((self.masks[name] + self.trunc_masks[name]) > 0).float()
        print('Total Model parameters:', total_size)
        density_size = 0
        trunc_density_size = 0
        for name, weight in self.masks.items():
            density_size += (weight != 0).sum().int().item()
            trunc_density_size += (self.trunc_masks[name] != 0).sum().int().item()
        print('Total parameters under density level of {:2f}%: {:2f}%, Searched {:2f}%'.format(100*self.density, 100*density_size / total_size, 100*trunc_density_size / total_size))

    def check_weight(self):
        all_num1 = 0
        all_num2 = 0
        for name, weight in self.model.named_parameters():
            name_mask = name + '_mask'
            num1 = ((weight + self.masks[name_mask]) == 0).sum().int().item()
            all_num1 += num1
            num2 = (self.masks[name_mask] == 0).sum().int().item()
            all_num2 += num2 
        print('Check Mask = {}'.format(all_num1 == all_num2))

