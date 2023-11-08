import os 
import time 
import argparse
import numpy as np 
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

import torchquantum as tq
import torchquantum.functional as tqf
from examples.core.datasets import MNIST
from examples.core.schedulers import CosineAnnealingWarmupRestarts

from dst import *
from utils import *

# noise_model
from torchquantum.plugins import QiskitProcessor
from torchquantum.plugins import tq2qiskit, qiskit2tq
from torchquantum.noise_model import *
from torchquantum.utils import (build_module_from_op_list,
                                build_module_op_list,
                                get_v_c_reg_mapping,
                                get_p_c_reg_mapping,
                                get_p_v_reg_mapping,
                                get_cared_configs)

# vowel digit

def main():
    parser = argparse.ArgumentParser()
    ############## Data setting ##############################
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--input_name', type=str, default='image')
    parser.add_argument('--target_name', type=str, default='digit')
    parser.add_argument('--num_class', type=int, default=4)
    parser.add_argument('--n_train_images', type=int, default=None)
    parser.add_argument('--n_val_images', type=int, default=None)
    parser.add_argument('--n_test_images', type=int, default=None)
    ############## Model setting ##############################
    parser.add_argument('--arch', type=str, default=None)
    parser.add_argument('--n_wires', type=int, default=4)
    parser.add_argument('--n_blocks', type=int, default=3)
    parser.add_argument('--save_dir', type=str, default=None)
    ############## Train setting ##############################
    parser.add_argument('--epochs', type=int, default=50, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=300)  
    parser.add_argument('--lr', type=float, default=0.3)
    parser.add_argument('--weight_decay', type=float, default=0)  
    parser.add_argument('--min_lr', type=float, default=0.03)  
    parser.add_argument('--warmup_epochs', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    ############## Device setting ##############################
    parser.add_argument('--qc_device', type=str, default='ibmq_santiago')
    parser.add_argument('--hub', type=str, default=None)
    parser.add_argument('--use_qc', action='store_true')
    parser.add_argument('--n_shots', type=int, default=8192)
    parser.add_argument('--optimization_level', type=int, default=2)
    parser.add_argument('--max_job', type=int, default=1)
    ############## Dynamic sparse training #######################
    parser.add_argument('--death_mode', type=str, default='magnitude')
    parser.add_argument('--growth_mode', type=str, default='gradient_prob')
    parser.add_argument('--init_mode', type=str, default='uniform_global')
    parser.add_argument('--density', type=float, default=0.5)
    parser.add_argument('--death_rate', type=float, default=0.4)    
    parser.add_argument('--frequency', type=int, default=5)
    parser.add_argument('--grad_beta', type=float, default=0.9)  
    parser.add_argument('--apply_sign', action='store_true')  
    parser.add_argument('--fix_arch', action='store_true')  
    parser.add_argument('--cosine_decay', action='store_true') 
    parser.add_argument('--t_end', type=int, default=35)
    ############## Optional setting ##############################
    parser.add_argument('--static', action='store_true', help='compute with static mode')
    parser.add_argument('--wires-per-block', type=int, default=2, help='wires per block int static mode')
    args = parser.parse_args()
    beta_acc = 0
    print(args)

    setup_seed(args.seed)

    # set-up devices
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    # dataset-Model
    dataflow = get_dataflow(args)
    model = eval(args.arch)(n_wires=args.n_wires, num_class=args.num_class, n_blocks=args.n_blocks)


    print('Model = {}'.format(args.arch))
    processor = QiskitProcessor(
        use_real_qc=args.use_qc,
        backend_name=args.qc_device,
        noise_model_name=args.qc_device,
        coupling_map_name=args.qc_device,
        basis_gates_name=args.qc_device,
        n_shots=args.n_shots,
        initial_layout=None,
        optimization_level=args.optimization_level,
        max_jobs=args.max_job,
        remove_ops_thres=None,
        transpile_with_ancilla=False,
        seed_transpiler=args.seed,
        seed_simulator=args.seed,
        hub=args.hub
    )

    model.set_qiskit_processor(processor)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=args.epochs,
        max_lr=args.lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_epochs,
    )

    decay = CosineDecay(args.death_rate, len(dataflow['train']) * (args.epochs))
    masker = Masking(
        optimizer = optimizer,
        death_rate=args.death_rate,
        death_rate_decay=decay, 
        death_mode=args.death_mode, 
        growth_mode=args.growth_mode,
        args=args,
        decay_schedule='cosine'
    )

    masker.add_module(model, args.density, reset_operation=True, sparse_init=args.init_mode)

    if args.static:
        model.q_layer.static_on(wires_per_block=args.wires_per_block)

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}:")

        # train
        masker.check_weight()
        scheduler.step()
        start = time.time()        
        train_and_return_grad(dataflow, model, device, optimizer, args, masker, gradient_beta=args.grad_beta, skip_ops=True, qiskit=True)
        print('Learning rate = {}'.format(optimizer.param_groups[0]['lr']))
        print('Time for 1 epoch = {:.2f}'.format(time.time()-start))

        # valid
        val_acc = valid_test_v2(dataflow, 'valid', model, device, args, qiskit=False)
        if not args.use_qc:
            val_acc_qiskit = valid_test_v2(dataflow, 'valid', model, device, args, qiskit=True)
            is_best_acc = val_acc_qiskit > beta_acc
            beta_acc = max(val_acc_qiskit, beta_acc)
        else:
            is_best_acc = val_acc > beta_acc
            beta_acc = max(val_acc, beta_acc)

        if is_best_acc:
            test_acc = valid_test_v2(dataflow, 'test', model, device, args, qiskit=False)
            if not args.use_qc:
                test_acc_qiskit = valid_test_v2(dataflow, 'test', model, device, args, qiskit=True)
                print('Best Test Accuracy = {:.4f}, Qiskit = {:.4f}'.format(100 * test_acc, 100 * test_acc_qiskit))
            else:
                print('Best Test Accuracy = {:.4f}'.format(100 * test_acc))

            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pt'))
            print('Best Epoch = {}'.format(epoch))
        torch.save(model.state_dict(), os.path.join(args.save_dir, 'checkpoint.pt'))
        torch.save({
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'best_acc': beta_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, os.path.join(args.save_dir, 'resume.pth'))

        # update structure
        if args.density < 1 and not args.fix_arch:
            if epoch % args.frequency == 0 and epoch <= args.t_end:
                if args.cosine_decay:
                    beta_random = growth_rand_beta_cosine(epoch, args.t_end)
                else:
                    beta_random = growth_rand_beta(epoch, args.t_end)
                masker.update_structure(beta_random, reset_operation=True)

    # Evaluate the best Model
    processor_real_qc = QiskitProcessor(
        use_real_qc=True,
        backend_name=args.qc_device,
        noise_model_name=args.qc_device,
        coupling_map_name=args.qc_device,
        basis_gates_name=args.qc_device,
        n_shots=args.n_shots,
        initial_layout=None,
        optimization_level=args.optimization_level,
        max_jobs=args.max_job,
        remove_ops_thres=None,
        transpile_with_ancilla=False,
        seed_transpiler=args.seed,
        seed_simulator=args.seed,
        hub=args.hub
    )
    best_checkpoint = torch.load(os.path.join(args.save_dir, 'checkpoint.pt'))
    model.load_state_dict(best_checkpoint)
    model.set_qiskit_processor(processor_real_qc)
    # check sparsity
    test_acc = valid_test_v2(dataflow, 'test', model, device, args, qiskit=False)
    test_acc_qiskit = valid_test_v2(dataflow, 'test', model, device, args, qiskit=True)
    print('Final Test Accuracy = {:.4f}, Qiskit = {:.4f}'.format(100 * test_acc, 100 * test_acc_qiskit))
    check_sparsity(model)

if __name__ == '__main__':
    main()
