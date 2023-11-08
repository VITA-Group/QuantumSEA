import torch.nn as nn
import torchquantum as tq
import torch.nn.utils.prune

from typing import Any, Callable, Dict
from torchpack.train import Trainer
from torchpack.utils.typing import Optimizer, Scheduler
from torchpack.utils.config import configs
from torchquantum.utils import (get_unitary_loss, legalize_unitary,
                                build_module_op_list)
from torchquantum.super_utils import ArchSampler
from torchquantum.prune_utils import (PhaseL1UnstructuredPruningMethod,
                                      ThresholdScheduler)
from torchpack.utils.logging import logger
from torchpack.callbacks.writers import TFEventWriter
from examples.core.tools.quantize import PACTActivationQuantizer
from torchpack.train.exception import StopTraining

__all__ = ['QTrainer', 'LayerRegressionTrainer', 'SuperQTrainer', 'ParamsShiftTrainer',
           'PruningTrainer', 'QNoiseAwareTrainer']


class LayerRegressionTrainer(Trainer):
    def __init__(self, *, model: nn.Module, criterion: Callable,
                 optimizer: Optimizer, scheduler: Scheduler) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    def _before_epoch(self) -> None:
        self.model.train()

    def _run_step(self, feed_dict: Dict[str, Any]) -> Dict[str, Any]:
        if configs.run.device == 'gpu':
            inputs = feed_dict['input'].cuda(non_blocking=True)
            targets = feed_dict['output'].cuda(non_blocking=True)
        else:
            inputs = feed_dict['input']
            targets = feed_dict['output']

        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)

        if loss.requires_grad:
            self.summary.add_scalar('loss', loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return {'outputs': outputs, 'targets': targets}

    def _after_epoch(self) -> None:
        self.model.eval()
        self.scheduler.step()

    def _state_dict(self) -> Dict[str, Any]:
        state_dict = dict()
        state_dict['model'] = self.model.state_dict()
        state_dict['optimizer'] = self.optimizer.state_dict()
        state_dict['scheduler'] = self.scheduler.state_dict()
        return state_dict

    def _load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])


class QTrainer(Trainer):
    def __init__(self, *, model: nn.Module, criterion: Callable,
                 optimizer: Optimizer, scheduler: Scheduler) -> None:
        self.model = model
        self.legalized_model = None
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.solution = None
        self.score = None

    def _before_epoch(self) -> None:
        self.model.train()

    def run_step(self, feed_dict: Dict[str, Any], legalize=False) -> Dict[
            str, Any]:
        output_dict = self._run_step(feed_dict, legalize=legalize)
        return output_dict

    def _run_step(self, feed_dict: Dict[str, Any], legalize=False) -> Dict[
            str, Any]:
        if configs.run.device == 'gpu':
            inputs = feed_dict[configs.dataset.input_name].cuda(
                non_blocking=True)
            targets = feed_dict[configs.dataset.target_name].cuda(
                non_blocking=True)
        else:
            inputs = feed_dict[configs.dataset.input_name]
            targets = feed_dict[configs.dataset.target_name]
        if legalize:
            outputs = self.legalized_model(inputs)
        else:
            outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        nll_loss = loss.item()
        unitary_loss = 0

        if configs.regularization.unitary_loss:
            unitary_loss = get_unitary_loss(self.model)
            if configs.regularization.unitary_loss_lambda_trainable:
                loss += self.model.unitary_loss_lambda[0] * unitary_loss
            else:
                loss += configs.regularization.unitary_loss_lambda * \
                        unitary_loss

        if loss.requires_grad:
            for k, group in enumerate(self.optimizer.param_groups):
                self.summary.add_scalar(f'lr/lr_group{k}', group['lr'])
            self.summary.add_scalar('loss', loss.item())
            self.summary.add_scalar('nll_loss', nll_loss)
            if getattr(self.model, 'sample_arch', None) is not None:
                for writer in self.summary.writers:
                    if isinstance(writer, TFEventWriter):
                        writer.writer.add_text(
                            'sample_arch', str(self.model.sample_arch),
                            self.global_step)

            if configs.regularization.unitary_loss:
                if configs.regularization.unitary_loss_lambda_trainable:
                    self.summary.add_scalar(
                        'u_loss_lambda',
                        self.model.unitary_loss_lambda.item())
                else:
                    self.summary.add_scalar(
                        'u_loss_lambda',
                        configs.regularization.unitary_loss_lambda)
                self.summary.add_scalar('u_loss', unitary_loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return {'outputs': outputs, 'targets': targets}

    def _after_epoch(self) -> None:
        self.model.eval()
        self.scheduler.step()
        if configs.legalization.legalize:
            if self.epoch_num % configs.legalization.epoch_interval == 0:
                legalize_unitary(self.model)

    def _after_step(self, output_dict) -> None:
        if configs.legalization.legalize:
            if self.global_step % configs.legalization.step_interval == 0:
                legalize_unitary(self.model)

    def _state_dict(self) -> Dict[str, Any]:
        state_dict = dict()
        # need to store model arch because of randomness of random layers
        state_dict['model_arch'] = self.model
        state_dict['model'] = self.model.state_dict()
        state_dict['optimizer'] = self.optimizer.state_dict()
        state_dict['scheduler'] = self.scheduler.state_dict()
        if getattr(self.model, 'sample_arch', None) is not None:
            state_dict['sample_arch'] = self.model.sample_arch

        if self.solution is not None:
            state_dict['solution'] = self.solution
            state_dict['score'] = self.score

        if getattr(self.model, 'encoder', None) is not None:
            if getattr(self.model.encoder, 'func_list', None) is not None:
                state_dict['encoder_func_list'] = self.model.encoder.func_list

        if getattr(self.model, 'q_layer', None) is not None:
            state_dict['q_layer_op_list'] = build_module_op_list(
                self.model.q_layer)

        if getattr(self.model, 'measure', None) is not None:
            if getattr(self.model.measure,
                       'v_c_reg_mapping', None) is not None:
                state_dict['v_c_reg_mapping'] = \
                    self.model.measure.v_c_reg_mapping

        if getattr(self.model, 'nodes', None) is not None:
            state_dict['encoder_func_list'] = [
                node.encoder.func_list for node in self.model.nodes]
            if configs.model.transpile_before_run:
                # only save op_list if transpile before run
                state_dict['q_layer_op_list'] = [
                    build_module_op_list(node.q_layer) for node in
                    self.model.nodes]
            state_dict['v_c_reg_mapping'] = [
                node.measure.v_c_reg_mapping for node in self.model.nodes]
        for attr in ['v_c_reg_mapping', 'encoder_func_list',
                     'q_layer_op_list']:
            if state_dict.get(attr, None) is None:
                logger.warning(f"No {attr} found, will not save it.")

        return state_dict

    def _load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        # self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])


class ParamsShiftTrainer(Trainer):
    def __init__(self, *, model: nn.Module, criterion: Callable,
                 optimizer: Optimizer, scheduler: Scheduler) -> None:
        self.model = model
        self.legalized_model = None
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.solution = None
        self.score = None
        self.is_training = False
        self.save_global_step = 0
        self.quantizers = []
        self.init_act_quant()
        self.max_acc = 0
        self.max_acc_epoch = 0
        self.global_epoch = 0
        self.work_from_step = 0
        self.grad_dict = None
        self.processor_list = []
        self.profile_shots = False
        self.per_steps = 100
        self.record = dict()

    def load_grad(self, stop_step, grad_dict):
        self.work_from_step = stop_step
        self.grad_dict = grad_dict

    def set_processor_list(self, processor_list):
        self.processor_list = processor_list
        self.profile_shots = True
        self.per_steps = 25

    def init_act_quant(self):
        device = torch.device('cuda') if configs.run.device == 'gpu' else \
            torch.device('cpu')
        if configs.trainer.act_quant:
            quantizers = []
            assert getattr(self.model, 'nodes', None) is not None
            # single Q node
            for k, node in enumerate(self.model.nodes):
                # optionally skip the quantization of last layer output
                if configs.trainer.act_quant_skip_last_node and k == len(
                        self.model.nodes) - 1:
                    continue
                quantizer = PACTActivationQuantizer(
                    module=node,
                    precision=getattr(configs.trainer, 'act_quant_bit', None),
                    level=getattr(configs.trainer, 'act_quant_level', None),
                    alpha=1.0,
                    backprop_alpha=False,
                    quant_ratio=configs.trainer.act_quant_ratio,
                    device=device,
                    lower_bound=configs.trainer.act_quant_lower_bound,
                    upper_bound=configs.trainer.act_quant_upper_bound,
                )
                quantizers.append(quantizer)

            self.quantizers = quantizers

    def register_act_quant_hook(self):
        for quantizer in self.quantizers:
            quantizer.register_hook()

    def remove_act_quant_hook(self):
        for quantizer in self.quantizers:
            quantizer.remove_hook()

    def set_use_qiskit(self, configs):
        self.use_qiskit_train = configs.qiskit.use_qiskit_train
        self.use_qiskit_valid = configs.qiskit.use_qiskit_valid
    
    def _before_epoch(self) -> None:
        self.model.train()
        self.is_training = True
        self.register_act_quant_hook()

    def run_step(self, feed_dict: Dict[str, Any], legalize=False) -> Dict[
            str, Any]:
        if self.profile_shots and self.global_step % self.per_steps == 0 and self.is_training:
            self.record[self.global_step] = {}
            for i in range(4):
                for shots, processor in self.processor_list:
                    self.model.set_qiskit_processor(processor)
                    self.record[self.global_step][i] = self._get_grad(feed_dict, use_qiskit=True)
            output_dict = self._run_params_shift_step(feed_dict, use_qiskit=False)
            grad_list = []
            for param in self.model.parameters():
                grad_list.append(param.grad.item())
            self.record[self.global_step]['classical'] = grad_list

            return output_dict
        
        if self.is_training:
            output_dict = self._run_params_shift_step(feed_dict, use_qiskit=self.use_qiskit_train)
        else:
            output_dict = self._run_step(feed_dict, legalize=legalize, use_qiskit=self.use_qiskit_valid)
        return output_dict

    def _run_params_shift_step(self, feed_dict: Dict[str, Any], use_qiskit=False) -> Dict[
            str, Any]:
        if configs.run.device == 'gpu':
            inputs = feed_dict[configs.dataset.input_name].cuda(
                non_blocking=True)
            targets = feed_dict[configs.dataset.target_name].cuda(
                non_blocking=True)
        else:
            inputs = feed_dict[configs.dataset.input_name]
            targets = feed_dict[configs.dataset.target_name]
        
        # outputs = self.model(inputs)
        # loss = self.criterion(outputs, targets)
        # self.optimizer.zero_grad()
        # loss.backward()
        # for param in self.model.parameters():
        #     print(param.grad)

        outputs = None
        if self.global_step > self.work_from_step:
            outputs = self.model.shift_and_run(inputs, self.global_step,
                self.steps_per_epoch * self.num_epochs, verbose=False, use_qiskit=use_qiskit)
        else:
            logger.info('global_step {0}, skip parameters shift.'.format(self.global_step))
            outputs = self.model(inputs, False, use_qiskit=use_qiskit)
        loss = self.criterion(outputs, targets)
        nll_loss = loss.item()

        for k, group in enumerate(self.optimizer.param_groups):
            self.summary.add_scalar(f'lr/lr_group{k}', group['lr'])
        self.summary.add_scalar('loss', loss.item())
        self.summary.add_scalar('nll_loss', nll_loss)

        if self.global_step > self.work_from_step:
            self.optimizer.zero_grad()
            loss.backward()
            self.model.backprop_grad()
        else:
            logger.info('global_step {0}, load grad from tensorboard file!'.format(self.global_step))
            for i, param in enumerate(self.model.parameters()):
                param.grad = torch.tensor(self.grad_dict[self.global_step][i]).to(dtype=torch.float32, device=param.device).view(param.shape)
        # for param in self.model.parameters():
        #     print(param.grad)
        self.optimizer.step()
        # with torch.no_grad():
        #     for param in self.model.parameters():
        #         param.copy_(param - param.grad * 1.0)

        return {'outputs': outputs, 'targets': targets}


    def _get_grad(self, feed_dict: Dict[str, Any], use_qiskit=False) -> Dict[
            str, Any]:
        if configs.run.device == 'gpu':
            inputs = feed_dict[configs.dataset.input_name].cuda(
                non_blocking=True)
            targets = feed_dict[configs.dataset.target_name].cuda(
                non_blocking=True)
        else:
            inputs = feed_dict[configs.dataset.input_name]
            targets = feed_dict[configs.dataset.target_name]
        
        # outputs = self.model(inputs)
        # loss = self.criterion(outputs, targets)
        # self.optimizer.zero_grad()
        # loss.backward()
        # for param in self.model.parameters():
        #     print(param.grad)

        outputs = None
        if self.global_step > self.work_from_step:
            outputs = self.model.shift_and_run(inputs, self.global_step,
                self.steps_per_epoch * self.num_epochs, verbose=False, use_qiskit=use_qiskit)
        else:
            logger.info('global_step {0}, skip parameters shift.'.format(self.global_step))
            outputs = self.model(inputs, False, use_qiskit)
        loss = self.criterion(outputs, targets)
        nll_loss = loss.item()

        for k, group in enumerate(self.optimizer.param_groups):
            self.summary.add_scalar(f'lr/lr_group{k}', group['lr'])
        self.summary.add_scalar('loss', loss.item())
        self.summary.add_scalar('nll_loss', nll_loss)

        self.optimizer.zero_grad()
        loss.backward()
        if self.global_step > self.work_from_step:
            self.model.backprop_grad()
        else:
            logger.info('global_step {0}, load grad from tensorboard file!'.format(self.global_step))
            for i, param in enumerate(self.model.parameters()):
                param.grad = torch.tensor(self.grad_dict[self.global_step][i]).to(dtype=torch.float32, device=param.device).view(param.shape)
        grad_list = []
        for param in self.model.parameters():
            grad_list.append(param.grad.item())
        # self.optimizer.step()
        # with torch.no_grad():
        #     for param in self.model.parameters():
        #         param.copy_(param - param.grad * 1.0)

        return grad_list



    def _run_step(self, feed_dict: Dict[str, Any], legalize=False, use_qiskit=False) -> Dict[
            str, Any]:
        if configs.run.device == 'gpu':
            inputs = feed_dict[configs.dataset.input_name].cuda(
                non_blocking=True)
            targets = feed_dict[configs.dataset.target_name].cuda(
                non_blocking=True)
        else:
            inputs = feed_dict[configs.dataset.input_name]
            targets = feed_dict[configs.dataset.target_name]
        if legalize:
            outputs = self.legalized_model(inputs)
        else:
            outputs = self.model(inputs, legalize, use_qiskit)
        loss = self.criterion(outputs, targets)
        nll_loss = loss.item()
        unitary_loss = 0

        if configs.regularization.unitary_loss:
            unitary_loss = get_unitary_loss(self.model)
            if configs.regularization.unitary_loss_lambda_trainable:
                loss += self.model.unitary_loss_lambda[0] * unitary_loss
            else:
                loss += configs.regularization.unitary_loss_lambda * \
                        unitary_loss

        if loss.requires_grad:
            for k, group in enumerate(self.optimizer.param_groups):
                self.summary.add_scalar(f'lr/lr_group{k}', group['lr'])
            self.summary.add_scalar('loss', loss.item())
            self.summary.add_scalar('nll_loss', nll_loss)
            if getattr(self.model, 'sample_arch', None) is not None:
                for writer in self.summary.writers:
                    if isinstance(writer, TFEventWriter):
                        writer.writer.add_text(
                            'sample_arch', str(self.model.sample_arch),
                            self.global_step)

            if configs.regularization.unitary_loss:
                if configs.regularization.unitary_loss_lambda_trainable:
                    self.summary.add_scalar(
                        'u_loss_lambda',
                        self.model.unitary_loss_lambda.item())
                else:
                    self.summary.add_scalar(
                        'u_loss_lambda',
                        configs.regularization.unitary_loss_lambda)
                self.summary.add_scalar('u_loss', unitary_loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return {'outputs': outputs, 'targets': targets}

    def _after_epoch(self) -> None:
        self.model.eval()
        self.is_training = False
        self.scheduler.step()
        if configs.legalization.legalize:
            if self.epoch_num % configs.legalization.epoch_interval == 0:
                legalize_unitary(self.model)
        self.save_global_step = self.global_step
        self.global_step = int(self.model.num_forwards)
    
    def _trigger_epoch(self) -> None:
        self.global_step = self.save_global_step

    def _after_step(self, output_dict) -> None:
        if configs.legalization.legalize:
            if self.global_step % configs.legalization.step_interval == 0:
                legalize_unitary(self.model)

    def _state_dict(self) -> Dict[str, Any]:
        self.remove_act_quant_hook()
        state_dict = dict()
        # need to store model arch because of randomness of random layers
        # state_dict['model_arch'] = self.model
        state_dict['model'] = self.model.state_dict()
        state_dict['optimizer'] = self.optimizer.state_dict()
        state_dict['scheduler'] = self.scheduler.state_dict()
        if getattr(self.model, 'sample_arch', None) is not None:
            state_dict['sample_arch'] = self.model.sample_arch

        if self.solution is not None:
            state_dict['solution'] = self.solution
            state_dict['score'] = self.score

        if getattr(self.model, 'encoder', None) is not None:
            if getattr(self.model.encoder, 'func_list', None) is not None:
                state_dict['encoder_func_list'] = self.model.encoder.func_list

        if getattr(self.model, 'q_layer', None) is not None:
            state_dict['q_layer_op_list'] = build_module_op_list(
                self.model.q_layer)

        if getattr(self.model, 'measure', None) is not None:
            if getattr(self.model.measure,
                       'v_c_reg_mapping', None) is not None:
                state_dict['v_c_reg_mapping'] = \
                    self.model.measure.v_c_reg_mapping
        
        if configs.trainer.act_quant:
            state_dict['act_quant'] = {
                'act_quant_bit': self.quantizers[0].precision,
                'act_quant_level': self.quantizers[0].level,
                'act_quant_ratio': self.quantizers[0].quant_ratio,
                'act_quant_lower_bound': self.quantizers[
                    0].lower_bound,
                'act_quant_upper_bound': self.quantizers[
                    0].upper_bound,
            }
        
        if getattr(self.model, 'nodes', None) is not None:
            state_dict['encoder_func_list'] = [
                node.encoder.func_list for node in self.model.nodes]
            if configs.model.transpile_before_run:
                # only save op_list if transpile before run
                state_dict['q_layer_op_list'] = [
                    build_module_op_list(node.q_layer) for node in
                    self.model.nodes]
            state_dict['v_c_reg_mapping'] = [
                node.measure.v_c_reg_mapping for node in self.model.nodes]
        for attr in ['v_c_reg_mapping', 'encoder_func_list',
                     'q_layer_op_list']:
            if state_dict.get(attr, None) is None:
                logger.warning(f"No {attr} found, will not save it.")

        return state_dict

    def _load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        # self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])

    def update_accuracy(self, name, accuracy):
        if name == 'acc/test':
            self.global_epoch += 1
            if accuracy > self.max_acc:
                self.max_acc = accuracy
                self.max_acc_epoch = self.global_epoch
            elif self.global_epoch - self.max_acc_epoch >= 50:
                raise StopTraining('Early stop at epoch{0}, num inferences = {1}, global_step={2}'.format(self.global_epoch, self.global_step, self.save_global_step))

class SuperQTrainer(Trainer):
    def __init__(self, *, model: nn.Module, criterion: Callable,
                 optimizer: Optimizer, scheduler: Scheduler) -> None:
        self.model = model
        self.legalized_model = None
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.sample_arch = None
        self.arch_sampler = ArchSampler(
            model=model,
            strategy=configs.model.sampler.strategy.dict(),
            n_layers_per_block=configs.model.arch.n_layers_per_block
        )

    def _before_epoch(self) -> None:
        self.model.train()
        self.arch_sampler.set_total_steps(self.steps_per_epoch *
                                          self.num_epochs)

    def _before_step(self, feed_dict: Dict[str, Any]) -> None:
        self.sample_arch = self.arch_sampler.get_uniform_sample_arch()
        self.model.set_sample_arch(self.sample_arch)

    def run_step(self, feed_dict: Dict[str, Any], legalize=False) -> Dict[
            str, Any]:
        output_dict = self._run_step(feed_dict, legalize=legalize)
        return output_dict

    def _run_step(self, feed_dict: Dict[str, Any], legalize=False) -> Dict[
            str, Any]:
        if configs.run.device == 'gpu':
            inputs = feed_dict[configs.dataset.input_name].cuda(
                non_blocking=True)
            targets = feed_dict[configs.dataset.target_name].cuda(
                non_blocking=True)
        else:
            inputs = feed_dict[configs.dataset.input_name]
            targets = feed_dict[configs.dataset.target_name]
        if legalize:
            outputs = self.legalized_model(inputs)
        else:
            outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        # nll_loss = loss.item()
        unitary_loss = 0

        if configs.regularization.unitary_loss:
            unitary_loss = get_unitary_loss(self.model)
            if configs.regularization.unitary_loss_lambda_trainable:
                loss += self.model.unitary_loss_lambda[0] * unitary_loss
            else:
                loss += configs.regularization.unitary_loss_lambda * \
                        unitary_loss

        if loss.requires_grad:
            # during training
            for k, group in enumerate(self.optimizer.param_groups):
                self.summary.add_scalar(f'lr/lr_group{k}', group['lr'])
            self.summary.add_scalar('loss', loss.item())
            # self.summary.add_scalar('nll_loss', nll_loss)
            self.summary.add_scalar('sta',
                                    self.arch_sampler.current_stage)
            self.summary.add_scalar('chk',
                                    self.arch_sampler.current_chunk)
            self.summary.add_scalar('n_ops',
                                    self.arch_sampler.sample_n_ops)
            if getattr(self.model, 'sample_arch', None) is not None:
                for writer in self.summary.writers:
                    if isinstance(writer, TFEventWriter):
                        writer.writer.add_text(
                            'sample_arch', str(self.model.sample_arch),
                            self.global_step)

            if configs.regularization.unitary_loss:
                if configs.regularization.unitary_loss_lambda_trainable:
                    self.summary.add_scalar(
                        'u_loss_lambda',
                        self.model.unitary_loss_lambda.item())
                else:
                    self.summary.add_scalar(
                        'u_loss_lambda',
                        configs.regularization.unitary_loss_lambda)
                self.summary.add_scalar('u_loss', unitary_loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return {'outputs': outputs, 'targets': targets}

    def _after_epoch(self) -> None:
        self.model.eval()
        self.scheduler.step()
        if configs.legalization.legalize:
            if self.epoch_num % configs.legalization.epoch_interval == 0:
                legalize_unitary(self.model)

    def _after_step(self, output_dict) -> None:
        if configs.legalization.legalize:
            if self.global_step % configs.legalization.step_interval == 0:
                legalize_unitary(self.model)

    def _state_dict(self) -> Dict[str, Any]:
        state_dict = dict()
        # need to store model arch because of randomness of random layers
        state_dict['model_arch'] = self.model
        state_dict['model'] = self.model.state_dict()
        state_dict['optimizer'] = self.optimizer.state_dict()
        state_dict['scheduler'] = self.scheduler.state_dict()
        if getattr(self.model, 'sample_arch', None) is not None:
            state_dict['sample_arch'] = self.model.sample_arch
        try:
            state_dict['v_c_reg_mapping'] = self.model.measure.v_c_reg_mapping
        except AttributeError:
            logger.warning(f"No v_c_reg_mapping found, will not save it.")
        return state_dict

    def _load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        # self.model.load_state_dict(state_dict['model'], strict=False)
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])


class PruningTrainer(Trainer):
    """
    Perform pruning-aware training
    """
    def __init__(self, *, model: nn.Module, criterion: Callable,
                 optimizer: Optimizer, scheduler: Scheduler) -> None:
        self.model = model
        self.legalized_model = None
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.solution = None
        self.score = None

        self._parameters_to_prune = None
        self._target_pruning_amount = None
        self._init_pruning_amount = None
        self.prune_amount_scheduler = None
        self.prune_amount = None

        self.init_pruning()

    @staticmethod
    def extract_prunable_parameters(model: nn.Module) -> list:
        _parameters_to_prune = [
            (module, "params")
            for _, module in model.named_modules() if isinstance(module,
                                                                 tq.Operator)
            and module.params is not None]
        return _parameters_to_prune

    def init_pruning(self) -> None:
        """
        Initialize pruning procedure
        """
        self._parameters_to_prune = self.extract_prunable_parameters(
            self.model)
        self._target_pruning_amount = configs.prune.target_pruning_amount
        self._init_pruning_amount = configs.prune.init_pruning_amount
        self.prune_amount_scheduler = ThresholdScheduler(
            configs.prune.start_epoch, configs.prune.end_epoch,
            self._init_pruning_amount,
            self._target_pruning_amount)
        self.prune_amount = self._init_pruning_amount

    def _remove_pruning(self):
        for module, name in self._parameters_to_prune:
            nn.utils.prune.remove(module, name)

    def _prune_model(self, prune_amount) -> None:
        """
        Perform global threshold/percentage pruning on the quantum model.
        This function just performs pruning re-parametrization, i.e.,
        record weight_orig and generate weight_mask
        """
        # first clear current pruning container, since we do not want cascaded
        # pruning methods
        # remove operation will make pruning permanent
        if self.epoch_num > 1:
            self._remove_pruning()
        # perform global phase pruning based on the given pruning amount
        nn.utils.prune.global_unstructured(
            self._parameters_to_prune,
            pruning_method=PhaseL1UnstructuredPruningMethod,
            amount=prune_amount,
        )
        self.summary.add_scalar('prune_amount', prune_amount)

    def _before_epoch(self) -> None:
        self.model.train()

    def run_step(self, feed_dict: Dict[str, Any], legalize=False) -> Dict[
            str, Any]:
        output_dict = self._run_step(feed_dict, legalize=legalize)
        return output_dict

    def _run_step(self, feed_dict: Dict[str, Any], legalize=False) -> Dict[
            str, Any]:
        if configs.run.device == 'gpu':
            inputs = feed_dict[configs.dataset.input_name].cuda(
                non_blocking=True)
            targets = feed_dict[configs.dataset.target_name].cuda(
                non_blocking=True)
        else:
            inputs = feed_dict[configs.dataset.input_name]
            targets = feed_dict[configs.dataset.target_name]
        if legalize:
            outputs = self.legalized_model(inputs)
        else:
            outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        nll_loss = loss.item()
        unitary_loss = 0

        if configs.regularization.unitary_loss:
            unitary_loss = get_unitary_loss(self.model)
            if configs.regularization.unitary_loss_lambda_trainable:
                loss += self.model.unitary_loss_lambda[0] * unitary_loss
            else:
                loss += configs.regularization.unitary_loss_lambda * \
                        unitary_loss

        if loss.requires_grad:
            for k, group in enumerate(self.optimizer.param_groups):
                self.summary.add_scalar(f'lr/lr_group{k}', group['lr'])

            self.summary.add_scalar('loss', loss.item())
            self.summary.add_scalar('nll_loss', nll_loss)
            if getattr(self.model, 'sample_arch', None) is not None:
                for writer in self.summary.writers:
                    if isinstance(writer, TFEventWriter):
                        writer.writer.add_text(
                            'sample_arch', str(self.model.sample_arch),
                            self.global_step)

            if configs.regularization.unitary_loss:
                if configs.regularization.unitary_loss_lambda_trainable:
                    self.summary.add_scalar(
                        'u_loss_lambda',
                        self.model.unitary_loss_lambda.item())
                else:
                    self.summary.add_scalar(
                        'u_loss_lambda',
                        configs.regularization.unitary_loss_lambda)
                self.summary.add_scalar('u_loss', unitary_loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return {'outputs': outputs, 'targets': targets}

    def _after_epoch(self) -> None:
        self.model.eval()
        self.scheduler.step()
        if configs.legalization.legalize:
            if self.epoch_num % configs.legalization.epoch_interval == 0:
                legalize_unitary(self.model)
        # update pruning amount using the scheduler
        self.prune_amount = self.prune_amount_scheduler.step()
        # prune the model
        self._prune_model(self.prune_amount)
        # commit pruned parameters after training
        if self.epoch_num == self.num_epochs:
            self._remove_pruning()

    def _after_step(self, output_dict) -> None:
        if configs.legalization.legalize:
            if self.global_step % configs.legalization.step_interval == 0:
                legalize_unitary(self.model)

    def _state_dict(self) -> Dict[str, Any]:
        state_dict = dict()
        # need to store model arch because of randomness of random layers
        state_dict['model_arch'] = self.model
        state_dict['model'] = self.model.state_dict()
        state_dict['optimizer'] = self.optimizer.state_dict()
        state_dict['scheduler'] = self.scheduler.state_dict()
        if getattr(self.model, 'sample_arch', None) is not None:
            state_dict['sample_arch'] = self.model.sample_arch
        try:
            state_dict['q_layer_op_list'] = build_module_op_list(
                self.model.q_layer)
            state_dict['encoder_func_list'] = self.model.encoder.func_list
        except AttributeError:
            logger.warning(f"No q_layer_op_list or encoder_func_list found, "
                           f"will not save them")

        if self.solution is not None:
            state_dict['solution'] = self.solution
            state_dict['score'] = self.score

        try:
            state_dict['v_c_reg_mapping'] = self.model.measure.v_c_reg_mapping
        except AttributeError:
            logger.warning(f"No v_c_reg_mapping found, will not save it.")
        return state_dict

    def _load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        # self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])


class QNoiseAwareTrainer(Trainer):
    def __init__(self, *, model: nn.Module, criterion: Callable,
                 optimizer: Optimizer, scheduler: Scheduler) -> None:
        self.model = model
        self.legalized_model = None
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.solution = None
        self.score = None
        self.quantizers = []
        self.init_act_quant()

    def init_act_quant(self):
        device = torch.device('cuda') if configs.run.device == 'gpu' else \
            torch.device('cpu')
        if configs.trainer.act_quant:
            quantizers = []
            assert getattr(self.model, 'nodes', None) is not None
            # single Q node
            for k, node in enumerate(self.model.nodes):
                # optionally skip the quantization of last layer output
                if configs.trainer.act_quant_skip_last_node and k == len(
                        self.model.nodes) - 1:
                    continue
                quantizer = PACTActivationQuantizer(
                    module=node,
                    precision=getattr(configs.trainer, 'act_quant_bit', None),
                    level=getattr(configs.trainer, 'act_quant_level', None),
                    alpha=1.0,
                    backprop_alpha=False,
                    quant_ratio=configs.trainer.act_quant_ratio,
                    device=device,
                    lower_bound=configs.trainer.act_quant_lower_bound,
                    upper_bound=configs.trainer.act_quant_upper_bound,
                )
                quantizers.append(quantizer)

            self.quantizers = quantizers

    def register_act_quant_hook(self):
        for quantizer in self.quantizers:
            quantizer.register_hook()

    def remove_act_quant_hook(self):
        for quantizer in self.quantizers:
            quantizer.remove_hook()

    def _before_epoch(self) -> None:
        self.model.train()
        # adjust the noise magnitude
        if self.model.noise_model_tq is not None:
            self.model.noise_model_tq.mode = 'train'
            self.model.noise_model_tq.adjust_noise(self.epoch_num)

        if getattr(self.model, 'nodes', None) is not None:
            for node in self.model.nodes:
                if node.noise_model_tq is not None:
                    node.noise_model_tq.mode = 'train'
                    node.noise_model_tq.adjust_noise(self.epoch_num)

        self.register_act_quant_hook()

    def run_step(self, feed_dict: Dict[str, Any], legalize=False) -> Dict[
            str, Any]:
        output_dict = self._run_step(feed_dict, legalize=legalize)
        return output_dict

    def _run_step(self, feed_dict: Dict[str, Any], legalize=False) -> Dict[
            str, Any]:
        if configs.run.device == 'gpu':
            inputs = feed_dict[configs.dataset.input_name].cuda(
                non_blocking=True)
            targets = feed_dict[configs.dataset.target_name].cuda(
                non_blocking=True)
        else:
            inputs = feed_dict[configs.dataset.input_name]
            targets = feed_dict[configs.dataset.target_name]
        if legalize:
            outputs = self.legalized_model(inputs)
        else:
            outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        nll_loss = loss.item()
        unitary_loss = 0

        if configs.regularization.unitary_loss:
            unitary_loss = get_unitary_loss(self.model)
            if configs.regularization.unitary_loss_lambda_trainable:
                loss += self.model.unitary_loss_lambda[0] * unitary_loss
            else:
                loss += configs.regularization.unitary_loss_lambda * \
                        unitary_loss

        if configs.trainer.act_quant and configs.trainer.act_quant_loss:
            mse_all = self.model.mse_all
            act_quant_loss = sum(mse_all)
            loss += configs.trainer.act_quant_loss_lambda * act_quant_loss

        if loss.requires_grad:
            for k, group in enumerate(self.optimizer.param_groups):
                self.summary.add_scalar(f'lr/lr_group{k}', group['lr'])
            self.summary.add_scalar('loss', loss.item())
            self.summary.add_scalar('nll_loss', nll_loss)

            if configs.trainer.act_quant and configs.trainer.act_quant_loss:
                self.summary.add_scalar('qloss', act_quant_loss.item())

            if self.model.noise_model_tq is not None:
                if self.model.noise_model_tq.noise_total_prob is not None:
                    noise_total_prob = \
                        self.model.noise_model_tq.noise_total_prob
                    if isinstance(noise_total_prob, list):
                        noise_total_prob = noise_total_prob[0]
                else:
                    noise_total_prob = -1
                self.summary.add_scalar('noip', noise_total_prob)
            if getattr(self.model, 'nodes', None) is not None:
                noise_total_prob = -1 if self.model.nodes[
                    0].noise_model_tq.noise_total_prob is None else \
                    self.model.nodes[0].noise_model_tq.noise_total_prob
                if isinstance(noise_total_prob, list):
                    noise_total_prob = noise_total_prob[0]
                self.summary.add_scalar('noip', noise_total_prob)

            if getattr(self.model, 'sample_arch', None) is not None:
                for writer in self.summary.writers:
                    if isinstance(writer, TFEventWriter):
                        writer.writer.add_text(
                            'sample_arch', str(self.model.sample_arch),
                            self.global_step)

            if configs.regularization.unitary_loss:
                if configs.regularization.unitary_loss_lambda_trainable:
                    self.summary.add_scalar(
                        'u_loss_lambda',
                        self.model.unitary_loss_lambda.item())
                else:
                    self.summary.add_scalar(
                        'u_loss_lambda',
                        configs.regularization.unitary_loss_lambda)
                self.summary.add_scalar('u_loss', unitary_loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return {'outputs': outputs, 'targets': targets}

    def _after_epoch(self) -> None:
        self.model.eval()
        self.scheduler.step()
        if configs.legalization.legalize:
            if self.epoch_num % configs.legalization.epoch_interval == 0:
                legalize_unitary(self.model)

        # set to eval mode, will not add noise
        if self.model.noise_model_tq is not None:
            self.model.noise_model_tq.mode = 'eval'
        if getattr(self.model, 'nodes', None) is not None:
            for node in self.model.nodes:
                if node.noise_model_tq is not None:
                    node.noise_model_tq.mode = 'eval'

    def _after_step(self, output_dict) -> None:
        if configs.legalization.legalize:
            if self.global_step % configs.legalization.step_interval == 0:
                legalize_unitary(self.model)

    def _state_dict(self) -> Dict[str, Any]:
        # remove hook otherwise the 'model' cannot be pickled
        self.remove_act_quant_hook()
        state_dict = dict()
        # need to store model arch because of randomness of random layers
        state_dict['model_arch'] = self.model
        state_dict['model'] = self.model.state_dict()
        state_dict['optimizer'] = self.optimizer.state_dict()
        state_dict['scheduler'] = self.scheduler.state_dict()
        if getattr(self.model, 'sample_arch', None) is not None:
            state_dict['sample_arch'] = self.model.sample_arch

        if self.solution is not None:
            state_dict['solution'] = self.solution
            state_dict['score'] = self.score

        if getattr(self.model, 'encoder', None) is not None:
            if getattr(self.model.encoder, 'func_list', None) is not None:
                state_dict['encoder_func_list'] = self.model.encoder.func_list

        if getattr(self.model, 'q_layer', None) is not None:
            state_dict['q_layer_op_list'] = build_module_op_list(
                self.model.q_layer)

        if getattr(self.model, 'measure', None) is not None:
            if getattr(self.model.measure,
                       'v_c_reg_mapping', None) is not None:
                state_dict['v_c_reg_mapping'] = \
                    self.model.measure.v_c_reg_mapping

        if getattr(self.model, 'noise_model_tq', None) is not None:
            state_dict['noise_model_tq'] = self.model.noise_model_tq

        if configs.trainer.act_quant:
            state_dict['act_quant'] = {
                'act_quant_bit': self.quantizers[0].precision,
                'act_quant_level': self.quantizers[0].level,
                'act_quant_ratio': self.quantizers[0].quant_ratio,
                'act_quant_lower_bound': self.quantizers[
                    0].lower_bound,
                'act_quant_upper_bound': self.quantizers[
                    0].upper_bound,
            }

        if getattr(self.model, 'nodes', None) is not None:
            state_dict['encoder_func_list'] = [
                node.encoder.func_list for node in self.model.nodes]
            if configs.model.transpile_before_run:
                # only save op_list if transpile before run
                state_dict['q_layer_op_list'] = [
                    build_module_op_list(node.q_layer) for node in
                    self.model.nodes]
            state_dict['v_c_reg_mapping'] = [
                node.measure.v_c_reg_mapping for node in self.model.nodes]

            # one node has one own noise model
            state_dict['noise_model_tq'] = [
                node.noise_model_tq for node in self.model.nodes]

        for attr in ['v_c_reg_mapping', 'encoder_func_list',
                     'q_layer_op_list', 'noise_model_tq']:
            if state_dict.get(attr, None) is None:
                logger.warning(f"No {attr} found, will not save it.")

        return state_dict

    def _load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        # self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])
