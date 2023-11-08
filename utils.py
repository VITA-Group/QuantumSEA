import pdb
import time 
import random
import argparse
import numpy as np 

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchquantum.vqe_utils import parse_hamiltonian_file

import torchquantum as tq
import torchquantum.functional as tqf

from examples.core.datasets import MNIST, Vowel, VQE
from examples.core.schedulers import CosineAnnealingWarmupRestarts
from dst import *


__all__ = ['U3CU3_Model', 'ZZRY_Model', 'RXYZ_Model', 'ZXXX_Model', 'RXYZ_U1_CU3_Model', 'IBMQ_Basis_Model',
        'U3CU3_Model_DST', 'ZZRY_Model_DST', 'RXYZ_Model_DST', 'ZXXX_Model_DST', 'RXYZ_U1_CU3_Model_DST', 'IBMQ_Basis_Model_DST', 'QFCModel_MNIST2_DST', 'QFCModel_MNIST4_DST',
        'U3CU3_Model_DST_VQE', 'ZZRY_Model_DST_VQE', 'RXYZ_Model_DST_VQE', 'ZXXX_Model_DST_VQE', 'RXYZ_U1_CU3_Model_DST_VQE', 'IBMQ_Basis_Model_DST_VQE',
        'U3CU3_Model_DST_Vowel', 'ZZRY_Model_DST_Vowel', 'RXYZ_Model_DST_Vowel', 'ZXXX_Model_DST_Vowel', 'RXYZ_U1_CU3_Model_DST_Vowel', 'IBMQ_Basis_Model_DST_Vowel',
        'get_dataflow', 'growth_rand_beta', 'growth_rand_beta_cosine', 'check_sparsity', 'setup_seed',
        'train', 'train_and_return_grad', 'valid_test', 'valid_test_v2', 'train_vqe', 'test_vqe', 'RXYZ_Model_DST_Quant', 'train_and_return_grad_quant', 'train_quant', 'RXYZ_Model_DST_Quant_GPU']


############## Model ############
class U3CU3_Model(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires, n_blocks):
            super().__init__()

            self.n_wires = n_wires
            self.n_blocks = n_blocks

            self.single_layers = tq.QuantumModuleList()
            self.control_layers = tq.QuantumModuleList()
            for _ in range(self.n_blocks):
                self.single_layers.append(
                    tq.Op1QAllLayer(op=tq.U3, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.control_layers.append(
                    tq.Op2QAllLayer(op=tq.CU3, n_wires=self.n_wires,
                                    has_params=True, trainable=True, circular=True))

        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
            self.q_device = q_device
            for k in range(self.n_blocks):
                self.single_layers[k](self.q_device)
                self.control_layers[k](self.q_device)

    def __init__(self, n_wires=4, num_class=4, n_blocks=8):
        super().__init__()
        self.n_wires = n_wires
        self.num_class = num_class
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict['4x4_ryzxy'])

        self.q_layer = self.QLayer(self.n_wires, n_blocks)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x, use_qiskit=False):
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6).view(bsz, 16)

        if use_qiskit:
            x = self.qiskit_processor.process_parameterized(
                self.q_device, self.encoder, self.q_layer, self.measure, x)
        else:
            self.encoder(self.q_device, x)
            self.q_layer(self.q_device)
            x = self.measure(self.q_device)

        x = x.reshape(bsz, 4)

        return x

class ZZRY_Model(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires, n_blocks):
            super().__init__()

            self.n_wires = n_wires
            self.n_blocks = n_blocks

            self.single_layers = tq.QuantumModuleList()
            self.control_layers = tq.QuantumModuleList()
            for _ in range(self.n_blocks):
                self.single_layers.append(
                    tq.Op1QAllLayer(op=tq.RY, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.control_layers.append(
                    tq.Op2QAllLayer(op=tq.RZZ, n_wires=self.n_wires,
                                    has_params=True, trainable=True, circular=True))
        
        # RZZ-Ring + RY
        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
            self.q_device = q_device
            for k in range(self.n_blocks):
                self.control_layers[k](self.q_device)
                self.single_layers[k](self.q_device)

    def __init__(self, n_wires=4, num_class=4, n_blocks=8):
        super().__init__()
        self.n_wires = n_wires
        self.num_class = num_class
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict['4x4_ryzxy'])

        self.q_layer = self.QLayer(self.n_wires, n_blocks)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x, use_qiskit=False):
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6).view(bsz, 16)

        if use_qiskit:
            x = self.qiskit_processor.process_parameterized(
                self.q_device, self.encoder, self.q_layer, self.measure, x)
        else:
            self.encoder(self.q_device, x)
            self.q_layer(self.q_device)
            x = self.measure(self.q_device)

        x = x.reshape(bsz, 4)

        return x

class RXYZ_Model(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires, n_blocks):
            super().__init__()

            self.n_wires = n_wires
            self.n_blocks = n_blocks
            
            self.hadamard = tq.Op1QAllLayer(op=tq.SHadamard, n_wires=self.n_wires)
            self.single_layers_1 = tq.QuantumModuleList()
            self.single_layers_2 = tq.QuantumModuleList()
            self.single_layers_3 = tq.QuantumModuleList()
            self.control_layers = tq.QuantumModuleList()
            for _ in range(self.n_blocks):
                self.single_layers_1.append(
                    tq.Op1QAllLayer(op=tq.RX, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.single_layers_2.append(
                    tq.Op1QAllLayer(op=tq.RY, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.single_layers_3.append(
                    tq.Op1QAllLayer(op=tq.RZ, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.control_layers.append(
                    tq.Op2QAllLayer(op=tq.CZ, n_wires=self.n_wires))
        
        # RX + RY + RZ + CZ
        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
            self.q_device = q_device
            self.hadamard(self.q_device)
            for k in range(self.n_blocks):
                self.single_layers_1[k](self.q_device)
                self.single_layers_2[k](self.q_device)
                self.single_layers_3[k](self.q_device)
                self.control_layers[k](self.q_device)

    def __init__(self, n_wires=4, num_class=4, n_blocks=8):
        super().__init__()
        self.n_wires = n_wires
        self.num_class = num_class
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict['4x4_ryzxy'])

        self.q_layer = self.QLayer(self.n_wires, n_blocks)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x, use_qiskit=False):
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6).view(bsz, 16)

        if use_qiskit:
            x = self.qiskit_processor.process_parameterized(
                self.q_device, self.encoder, self.q_layer, self.measure, x)
        else:
            self.encoder(self.q_device, x)
            self.q_layer(self.q_device)
            x = self.measure(self.q_device)

        x = x.reshape(bsz, 4)

        return x

class ZXXX_Model(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires, n_blocks):
            super().__init__()

            self.n_wires = n_wires
            self.n_blocks = n_blocks

            self.control_layers_1 = tq.QuantumModuleList()
            self.control_layers_2 = tq.QuantumModuleList()
            for _ in range(self.n_blocks):
                self.control_layers_1.append(
                    tq.Op2QAllLayer(op=tq.RZX, n_wires=self.n_wires,
                                    has_params=True, trainable=True, circular=True))
                self.control_layers_2.append(
                    tq.Op2QAllLayer(op=tq.RXX, n_wires=self.n_wires,
                                    has_params=True, trainable=True, circular=True))
        
        # RZX + RXX
        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
            self.q_device = q_device
            for k in range(self.n_blocks):
                self.control_layers_1[k](self.q_device)
                self.control_layers_2[k](self.q_device)

    def __init__(self, n_wires=4, num_class=4, n_blocks=8):
        super().__init__()
        self.n_wires = n_wires
        self.num_class = num_class
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict['4x4_ryzxy'])

        self.q_layer = self.QLayer(self.n_wires, n_blocks)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x, use_qiskit=False):
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6).view(bsz, 16)

        if use_qiskit:
            x = self.qiskit_processor.process_parameterized(
                self.q_device, self.encoder, self.q_layer, self.measure, x)
        else:
            self.encoder(self.q_device, x)
            self.q_layer(self.q_device)
            x = self.measure(self.q_device)

        x = x.reshape(bsz, 4)

        return x

class RXYZ_U1_CU3_Model(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires, n_blocks):
            super().__init__()

            self.n_wires = n_wires
            self.n_blocks = n_blocks

            self.single_layers_1 = tq.QuantumModuleList()
            self.single_layers_2 = tq.QuantumModuleList()
            self.single_layers_3 = tq.QuantumModuleList()
            self.single_layers_4 = tq.QuantumModuleList()
            self.single_layers_5 = tq.QuantumModuleList()
            self.single_layers_6 = tq.QuantumModuleList()
            self.single_layers_7 = tq.QuantumModuleList()

            self.control_layers_1 = tq.QuantumModuleList()
            self.control_layers_2 = tq.QuantumModuleList()
            self.control_layers_3 = tq.QuantumModuleList()
            self.control_layers_4 = tq.QuantumModuleList()

            for _ in range(self.n_blocks):
                self.single_layers_1.append(
                    tq.Op1QAllLayer(op=tq.RX, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.single_layers_2.append(
                    tq.Op1QAllLayer(op=tq.S, n_wires=self.n_wires))
                self.single_layers_3.append(
                    tq.Op1QAllLayer(op=tq.RY, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.single_layers_4.append(
                    tq.Op1QAllLayer(op=tq.T, n_wires=self.n_wires))
                self.single_layers_5.append(
                    tq.Op1QAllLayer(op=tq.RZ, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.single_layers_6.append(
                    tq.Op1QAllLayer(op=tq.T, n_wires=self.n_wires))
                self.single_layers_7.append(
                    tq.Op1QAllLayer(op=tq.U1, n_wires=self.n_wires,
                                    has_params=True, trainable=True))

                self.control_layers_1.append(
                    tq.Op2QAllLayer(op=tq.CNOT, n_wires=self.n_wires, circular=True))
                self.control_layers_2.append(
                    tq.Op2QAllLayer(op=tq.SWAP, n_wires=self.n_wires, circular=True))
                self.control_layers_3.append(
                    tq.Op2QAllLayer(op=tq.SSWAP, n_wires=self.n_wires, circular=True))
                self.control_layers_4.append(
                    tq.Op2QAllLayer(op=tq.CU3, n_wires=self.n_wires, 
                                    has_params=True, trainable=True, circular=True))

        # RXYZ + U1 + CU3
        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
            self.q_device = q_device
            for k in range(self.n_blocks):
                self.single_layers_1[k](self.q_device)
                self.single_layers_2[k](self.q_device)
                self.control_layers_1[k](self.q_device)
                self.single_layers_3[k](self.q_device)
                self.single_layers_4[k](self.q_device)
                self.control_layers_2[k](self.q_device)
                self.single_layers_5[k](self.q_device)
                self.single_layers_6[k](self.q_device)
                self.control_layers_3[k](self.q_device)
                self.single_layers_7[k](self.q_device)
                self.control_layers_4[k](self.q_device)

    def __init__(self, n_wires=4, num_class=4, n_blocks=4):
        super().__init__()
        self.n_wires = n_wires
        self.num_class = num_class
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict['4x4_ryzxy'])

        self.q_layer = self.QLayer(self.n_wires, n_blocks)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x, use_qiskit=False):
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6).view(bsz, 16)

        if use_qiskit:
            x = self.qiskit_processor.process_parameterized(
                self.q_device, self.encoder, self.q_layer, self.measure, x)
        else:
            self.encoder(self.q_device, x)
            self.q_layer(self.q_device)
            x = self.measure(self.q_device)

        x = x.reshape(bsz, 4)

        return x

class IBMQ_Basis_Model(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires, n_blocks):
            super().__init__()

            self.n_wires = n_wires
            self.n_blocks = n_blocks

            self.single_layers_1 = tq.QuantumModuleList()
            self.single_layers_2 = tq.QuantumModuleList()
            self.single_layers_3 = tq.QuantumModuleList()
            self.single_layers_4 = tq.QuantumModuleList()
            self.single_layers_5 = tq.QuantumModuleList()
            self.control_layers = tq.QuantumModuleList()

            for _ in range(self.n_blocks):
                self.single_layers_1.append(
                    tq.Op1QAllLayer(op=tq.RZ, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.single_layers_2.append(
                    tq.Op1QAllLayer(op=tq.PauliX, n_wires=self.n_wires))
                self.single_layers_3.append(
                    tq.Op1QAllLayer(op=tq.RZ, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.single_layers_4.append(
                    tq.Op1QAllLayer(op=tq.SX, n_wires=self.n_wires))
                self.single_layers_5.append(
                    tq.Op1QAllLayer(op=tq.RZ, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.control_layers.append(
                    tq.Op2QAllLayer(op=tq.CNOT, n_wires=self.n_wires, circular=True))

        # IBMQ-Basis
        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
            self.q_device = q_device
            for k in range(self.n_blocks):
                self.single_layers_1[k](self.q_device)
                self.single_layers_2[k](self.q_device)
                self.single_layers_3[k](self.q_device)
                self.single_layers_4[k](self.q_device)
                self.single_layers_5[k](self.q_device)
                self.control_layers[k](self.q_device)

    def __init__(self, n_wires=4, num_class=4, n_blocks=20):
        super().__init__()
        self.n_wires = n_wires
        self.num_class = num_class
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict['4x4_ryzxy'])

        self.q_layer = self.QLayer(self.n_wires, n_blocks)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x, use_qiskit=False):
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6).view(bsz, 16)

        if use_qiskit:
            x = self.qiskit_processor.process_parameterized(
                self.q_device, self.encoder, self.q_layer, self.measure, x)
        else:
            self.encoder(self.q_device, x)
            self.q_layer(self.q_device)
            x = self.measure(self.q_device)

        x = x.reshape(bsz, 4)

        return x

##  VQE ##
class U3CU3_Model_VQE(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires, n_blocks):
            super().__init__()

            self.n_wires = n_wires
            self.n_blocks = n_blocks

            self.single_layers = tq.QuantumModuleList()
            self.control_layers = tq.QuantumModuleList()
            for _ in range(self.n_blocks):
                self.single_layers.append(
                    tq.Op1QAllLayer(op=tq.U3, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.control_layers.append(
                    tq.Op2QAllLayer(op=tq.CU3, n_wires=self.n_wires,
                                    has_params=True, trainable=True, circular=True))

        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
            self.q_device = q_device
            for k in range(self.n_blocks):
                self.single_layers[k](self.q_device)
                self.control_layers[k](self.q_device)

    def __init__(self, hamil_info, n_wires=4, num_class=4, n_blocks=8):
        super().__init__()
        self.n_wires = n_wires
        self.hamil_info = hamil_info
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.q_layer = self.QLayer(self.n_wires, n_blocks)
        self.measure = tq.MeasureMultipleTimes(
            obs_list=hamil_info['hamil_list'])

    def forward(self, x, use_qiskit=False):
        bsz = x.shape[0]

        if use_qiskit:
            x = self.qiskit_processor.process_multi_measure(
                self.q_device, self.q_layer, self.measure)
        else:
            self.q_device.reset_states(bsz=1)
            self.q_layer(self.q_device)
            x = self.measure(self.q_device)

        hamil_coefficients = torch.tensor([hamil['coefficient'] for hamil in
                                        self.hamil_info['hamil_list']],
                                        device=x.device).double()

        for k, hamil in enumerate(self.hamil_info['hamil_list']):
            for wire, observable in zip(hamil['wires'], hamil['observables']):
                if observable == 'i':
                    x[k][wire] = 1
            for wire in range(self.q_device.n_wires):
                if wire not in hamil['wires']:
                    x[k][wire] = 1

        x = torch.cumprod(x, dim=-1)[:, -1].double()
        x = torch.dot(x, hamil_coefficients)

        if x.dim() == 0:
            x = x.unsqueeze(0)

        return x

class ZZRY_Model_VQE(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires, n_blocks):
            super().__init__()

            self.n_wires = n_wires
            self.n_blocks = n_blocks

            self.single_layers = tq.QuantumModuleList()
            self.control_layers = tq.QuantumModuleList()
            for _ in range(self.n_blocks):
                self.single_layers.append(
                    tq.Op1QAllLayer(op=tq.RY, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.control_layers.append(
                    tq.Op2QAllLayer(op=tq.RZZ, n_wires=self.n_wires,
                                    has_params=True, trainable=True, circular=True))
        
        # RZZ-Ring + RY
        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
            self.q_device = q_device
            for k in range(self.n_blocks):
                self.control_layers[k](self.q_device)
                self.single_layers[k](self.q_device)
    
    def __init__(self, hamil_info, n_wires=4, num_class=4, n_blocks=8):
        super().__init__()
        self.n_wires = n_wires
        self.hamil_info = hamil_info
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.q_layer = self.QLayer(self.n_wires, n_blocks)
        self.measure = tq.MeasureMultipleTimes(
            obs_list=hamil_info['hamil_list'])

    def forward(self, x, use_qiskit=False):
        bsz = x.shape[0]

        if use_qiskit:
            x = self.qiskit_processor.process_multi_measure(
                self.q_device, self.q_layer, self.measure)
        else:
            self.q_device.reset_states(bsz=1)
            self.q_layer(self.q_device)
            x = self.measure(self.q_device)

        hamil_coefficients = torch.tensor([hamil['coefficient'] for hamil in
                                        self.hamil_info['hamil_list']],
                                        device=x.device).double()

        for k, hamil in enumerate(self.hamil_info['hamil_list']):
            for wire, observable in zip(hamil['wires'], hamil['observables']):
                if observable == 'i':
                    x[k][wire] = 1
            for wire in range(self.q_device.n_wires):
                if wire not in hamil['wires']:
                    x[k][wire] = 1

        x = torch.cumprod(x, dim=-1)[:, -1].double()
        x = torch.dot(x, hamil_coefficients)

        if x.dim() == 0:
            x = x.unsqueeze(0)

        return x

class RXYZ_Model_VQE(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires, n_blocks):
            super().__init__()

            self.n_wires = n_wires
            self.n_blocks = n_blocks
            
            self.hadamard = tq.Op1QAllLayer(op=tq.SHadamard, n_wires=self.n_wires)
            self.single_layers_1 = tq.QuantumModuleList()
            self.single_layers_2 = tq.QuantumModuleList()
            self.single_layers_3 = tq.QuantumModuleList()
            self.control_layers = tq.QuantumModuleList()
            for _ in range(self.n_blocks):
                self.single_layers_1.append(
                    tq.Op1QAllLayer(op=tq.RX, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.single_layers_2.append(
                    tq.Op1QAllLayer(op=tq.RY, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.single_layers_3.append(
                    tq.Op1QAllLayer(op=tq.RZ, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.control_layers.append(
                    tq.Op2QAllLayer(op=tq.CZ, n_wires=self.n_wires))
        
        # RX + RY + RZ + CZ
        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
            self.q_device = q_device
            self.hadamard(self.q_device)
            for k in range(self.n_blocks):
                self.single_layers_1[k](self.q_device)
                self.single_layers_2[k](self.q_device)
                self.single_layers_3[k](self.q_device)
                self.control_layers[k](self.q_device)

    def __init__(self, hamil_info, n_wires=4, num_class=4, n_blocks=8):
        super().__init__()
        self.n_wires = n_wires
        self.hamil_info = hamil_info
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.q_layer = self.QLayer(self.n_wires, n_blocks)
        self.measure = tq.MeasureMultipleTimes(
            obs_list=hamil_info['hamil_list'])

    def forward(self, x, use_qiskit=False):
        bsz = x.shape[0]

        if use_qiskit:
            x = self.qiskit_processor.process_multi_measure(
                self.q_device, self.q_layer, self.measure)
        else:
            self.q_device.reset_states(bsz=1)
            self.q_layer(self.q_device)
            x = self.measure(self.q_device)

        hamil_coefficients = torch.tensor([hamil['coefficient'] for hamil in
                                        self.hamil_info['hamil_list']],
                                        device=x.device).double()

        for k, hamil in enumerate(self.hamil_info['hamil_list']):
            for wire, observable in zip(hamil['wires'], hamil['observables']):
                if observable == 'i':
                    x[k][wire] = 1
            for wire in range(self.q_device.n_wires):
                if wire not in hamil['wires']:
                    x[k][wire] = 1

        x = torch.cumprod(x, dim=-1)[:, -1].double()
        x = torch.dot(x, hamil_coefficients)

        if x.dim() == 0:
            x = x.unsqueeze(0)

        return x

class ZXXX_Model_VQE(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires, n_blocks):
            super().__init__()

            self.n_wires = n_wires
            self.n_blocks = n_blocks

            self.control_layers_1 = tq.QuantumModuleList()
            self.control_layers_2 = tq.QuantumModuleList()
            for _ in range(self.n_blocks):
                self.control_layers_1.append(
                    tq.Op2QAllLayer(op=tq.RZX, n_wires=self.n_wires,
                                    has_params=True, trainable=True, circular=True))
                self.control_layers_2.append(
                    tq.Op2QAllLayer(op=tq.RXX, n_wires=self.n_wires,
                                    has_params=True, trainable=True, circular=True))
        
        # RZX + RXX
        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
            self.q_device = q_device
            for k in range(self.n_blocks):
                self.control_layers_1[k](self.q_device)
                self.control_layers_2[k](self.q_device)

    def __init__(self, hamil_info, n_wires=4, num_class=4, n_blocks=8):
        super().__init__()
        self.n_wires = n_wires
        self.hamil_info = hamil_info
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.q_layer = self.QLayer(self.n_wires, n_blocks)
        self.measure = tq.MeasureMultipleTimes(
            obs_list=hamil_info['hamil_list'])

    def forward(self, x, use_qiskit=False):
        bsz = x.shape[0]

        if use_qiskit:
            x = self.qiskit_processor.process_multi_measure(
                self.q_device, self.q_layer, self.measure)
        else:
            self.q_device.reset_states(bsz=1)
            self.q_layer(self.q_device)
            x = self.measure(self.q_device)

        hamil_coefficients = torch.tensor([hamil['coefficient'] for hamil in
                                        self.hamil_info['hamil_list']],
                                        device=x.device).double()

        for k, hamil in enumerate(self.hamil_info['hamil_list']):
            for wire, observable in zip(hamil['wires'], hamil['observables']):
                if observable == 'i':
                    x[k][wire] = 1
            for wire in range(self.q_device.n_wires):
                if wire not in hamil['wires']:
                    x[k][wire] = 1

        x = torch.cumprod(x, dim=-1)[:, -1].double()
        x = torch.dot(x, hamil_coefficients)

        if x.dim() == 0:
            x = x.unsqueeze(0)

        return x

class RXYZ_U1_CU3_Model_VQE(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires, n_blocks):
            super().__init__()

            self.n_wires = n_wires
            self.n_blocks = n_blocks

            self.single_layers_1 = tq.QuantumModuleList()
            self.single_layers_2 = tq.QuantumModuleList()
            self.single_layers_3 = tq.QuantumModuleList()
            self.single_layers_4 = tq.QuantumModuleList()
            self.single_layers_5 = tq.QuantumModuleList()
            self.single_layers_6 = tq.QuantumModuleList()
            self.single_layers_7 = tq.QuantumModuleList()

            self.control_layers_1 = tq.QuantumModuleList()
            self.control_layers_2 = tq.QuantumModuleList()
            self.control_layers_3 = tq.QuantumModuleList()
            self.control_layers_4 = tq.QuantumModuleList()

            for _ in range(self.n_blocks):
                self.single_layers_1.append(
                    tq.Op1QAllLayer(op=tq.RX, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.single_layers_2.append(
                    tq.Op1QAllLayer(op=tq.S, n_wires=self.n_wires))
                self.single_layers_3.append(
                    tq.Op1QAllLayer(op=tq.RY, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.single_layers_4.append(
                    tq.Op1QAllLayer(op=tq.T, n_wires=self.n_wires))
                self.single_layers_5.append(
                    tq.Op1QAllLayer(op=tq.RZ, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.single_layers_6.append(
                    tq.Op1QAllLayer(op=tq.T, n_wires=self.n_wires))
                self.single_layers_7.append(
                    tq.Op1QAllLayer(op=tq.U1, n_wires=self.n_wires,
                                    has_params=True, trainable=True))

                self.control_layers_1.append(
                    tq.Op2QAllLayer(op=tq.CNOT, n_wires=self.n_wires, circular=True))
                self.control_layers_2.append(
                    tq.Op2QAllLayer(op=tq.SWAP, n_wires=self.n_wires, circular=True))
                self.control_layers_3.append(
                    tq.Op2QAllLayer(op=tq.SSWAP, n_wires=self.n_wires, circular=True))
                self.control_layers_4.append(
                    tq.Op2QAllLayer(op=tq.CU3, n_wires=self.n_wires, 
                                    has_params=True, trainable=True, circular=True))

        # RXYZ + U1 + CU3
        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
            self.q_device = q_device
            for k in range(self.n_blocks):
                self.single_layers_1[k](self.q_device)
                self.single_layers_2[k](self.q_device)
                self.control_layers_1[k](self.q_device)
                self.single_layers_3[k](self.q_device)
                self.single_layers_4[k](self.q_device)
                self.control_layers_2[k](self.q_device)
                self.single_layers_5[k](self.q_device)
                self.single_layers_6[k](self.q_device)
                self.control_layers_3[k](self.q_device)
                self.single_layers_7[k](self.q_device)
                self.control_layers_4[k](self.q_device)

    def __init__(self, hamil_info, n_wires=4, num_class=4, n_blocks=8):
        super().__init__()
        self.n_wires = n_wires
        self.hamil_info = hamil_info
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.q_layer = self.QLayer(self.n_wires, n_blocks)
        self.measure = tq.MeasureMultipleTimes(
            obs_list=hamil_info['hamil_list'])

    def forward(self, x, use_qiskit=False):
        bsz = x.shape[0]

        if use_qiskit:
            x = self.qiskit_processor.process_multi_measure(
                self.q_device, self.q_layer, self.measure)
        else:
            self.q_device.reset_states(bsz=1)
            self.q_layer(self.q_device)
            x = self.measure(self.q_device)

        hamil_coefficients = torch.tensor([hamil['coefficient'] for hamil in
                                        self.hamil_info['hamil_list']],
                                        device=x.device).double()

        for k, hamil in enumerate(self.hamil_info['hamil_list']):
            for wire, observable in zip(hamil['wires'], hamil['observables']):
                if observable == 'i':
                    x[k][wire] = 1
            for wire in range(self.q_device.n_wires):
                if wire not in hamil['wires']:
                    x[k][wire] = 1

        x = torch.cumprod(x, dim=-1)[:, -1].double()
        x = torch.dot(x, hamil_coefficients)

        if x.dim() == 0:
            x = x.unsqueeze(0)

        return x

class IBMQ_Basis_Model_VQE(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires, n_blocks):
            super().__init__()

            self.n_wires = n_wires
            self.n_blocks = n_blocks

            self.single_layers_1 = tq.QuantumModuleList()
            self.single_layers_2 = tq.QuantumModuleList()
            self.single_layers_3 = tq.QuantumModuleList()
            self.single_layers_4 = tq.QuantumModuleList()
            self.single_layers_5 = tq.QuantumModuleList()
            self.control_layers = tq.QuantumModuleList()

            for _ in range(self.n_blocks):
                self.single_layers_1.append(
                    tq.Op1QAllLayer(op=tq.RZ, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.single_layers_2.append(
                    tq.Op1QAllLayer(op=tq.PauliX, n_wires=self.n_wires))
                self.single_layers_3.append(
                    tq.Op1QAllLayer(op=tq.RZ, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.single_layers_4.append(
                    tq.Op1QAllLayer(op=tq.SX, n_wires=self.n_wires))
                self.single_layers_5.append(
                    tq.Op1QAllLayer(op=tq.RZ, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.control_layers.append(
                    tq.Op2QAllLayer(op=tq.CNOT, n_wires=self.n_wires, circular=True))

        # IBMQ-Basis
        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
            self.q_device = q_device
            for k in range(self.n_blocks):
                self.single_layers_1[k](self.q_device)
                self.single_layers_2[k](self.q_device)
                self.single_layers_3[k](self.q_device)
                self.single_layers_4[k](self.q_device)
                self.single_layers_5[k](self.q_device)
                self.control_layers[k](self.q_device)

    def __init__(self, hamil_info, n_wires=4, num_class=4, n_blocks=8):
        super().__init__()
        self.n_wires = n_wires
        self.hamil_info = hamil_info
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.q_layer = self.QLayer(self.n_wires, n_blocks)
        self.measure = tq.MeasureMultipleTimes(
            obs_list=hamil_info['hamil_list'])

    def forward(self, x, use_qiskit=False):
        bsz = x.shape[0]

        if use_qiskit:
            x = self.qiskit_processor.process_multi_measure(
                self.q_device, self.q_layer, self.measure)
        else:
            self.q_device.reset_states(bsz=1)
            self.q_layer(self.q_device)
            x = self.measure(self.q_device)

        hamil_coefficients = torch.tensor([hamil['coefficient'] for hamil in
                                        self.hamil_info['hamil_list']],
                                        device=x.device).double()

        for k, hamil in enumerate(self.hamil_info['hamil_list']):
            for wire, observable in zip(hamil['wires'], hamil['observables']):
                if observable == 'i':
                    x[k][wire] = 1
            for wire in range(self.q_device.n_wires):
                if wire not in hamil['wires']:
                    x[k][wire] = 1

        x = torch.cumprod(x, dim=-1)[:, -1].double()
        x = torch.dot(x, hamil_coefficients)

        if x.dim() == 0:
            x = x.unsqueeze(0)

        return x


##### DST, parameter shift #####
class Op1QAllLayer_DST(tq.QuantumModule):
    def __init__(self, op, n_wires: int, has_params=False, trainable=False):
        super().__init__()
        self.n_wires = n_wires
        self.op = op
        self.has_params = has_params
        self.trainable = trainable
        self.ops_all = tq.QuantumModuleList()

        self.register_buffer('op_flag', torch.ones(n_wires))
        for k in range(n_wires):
            self.ops_all.append(op(has_params=has_params, trainable=trainable))

    @tq.static_support
    def forward(self, q_device):
        for k in range(self.n_wires):
            if self.op_flag[k]:
                self.ops_all[k](q_device, wires=k)

    def reset_ops_sampling(self):
        if not (self.has_params and self.trainable): return 0
        with torch.no_grad():
            for k in range(self.n_wires):
                self.op_flag[k] = (self.ops_all[k].params.data.abs().sum() > 0).float()
        print('Update ops sampling: {}'.format(self.op_flag.data))

class Op2QAllLayer_DST(tq.QuantumModule):
    """pattern:
    circular = False
    jump = 1: [0, 1], [1, 2], [2, 3], [3, 4], [4, 5]
    jump = 2: [0, 2], [1, 3], [2, 4], [3, 5]
    jump = 3: [0, 3], [1, 4], [2, 5]
    jump = 4: [0, 4], [1, 5]
    jump = 5: [0, 5]

    circular = True
    jump = 1: [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]
    jump = 2: [0, 2], [1, 3], [2, 4], [3, 5], [4, 0], [5, 1]
    jump = 3: [0, 3], [1, 4], [2, 5], [3, 0], [4, 1], [5, 2]
    jump = 4: [0, 4], [1, 5], [2, 0], [3, 1], [4, 2], [5, 3]
    jump = 5: [0, 5], [1, 0], [2, 1], [3, 2], [4, 3], [5, 4]
    """
    def __init__(self, op, n_wires: int, has_params=False, trainable=False,
                 wire_reverse=False, jump=1, circular=False):
        super().__init__()
        self.n_wires = n_wires
        self.jump = jump
        self.circular = circular
        self.op = op
        self.has_params = has_params
        self.trainable = trainable
        self.ops_all = tq.QuantumModuleList()

        # reverse the wires, for example from [1, 2] to [2, 1]
        self.wire_reverse = wire_reverse

        if circular:
            n_ops = n_wires
        else:
            n_ops = n_wires - jump
        self.register_buffer('op_flag', torch.ones(n_ops))
        for k in range(n_ops):
            self.ops_all.append(op(has_params=has_params,
                                   trainable=trainable))

    @tq.static_support
    def forward(self, q_device):
        for k in range(len(self.ops_all)):
            if self.op_flag[k]:
                wires = [k, (k + self.jump) % self.n_wires]
                if self.wire_reverse:
                    wires.reverse()
                self.ops_all[k](q_device, wires=wires)

    def reset_ops_sampling(self):
        if not (self.has_params and self.trainable): return 0
        with torch.no_grad():
            for k in range(len(self.ops_all)):
                self.op_flag[k] = (self.ops_all[k].params.data.abs().sum() > 0).float()
        print('Update ops sampling: {}'.format(self.op_flag.data))

class U3CU3_Model_DST(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires, n_blocks):
            super().__init__()

            self.n_wires = n_wires
            self.n_blocks = n_blocks

            self.single_layers = tq.QuantumModuleList()
            self.control_layers = tq.QuantumModuleList()
            for _ in range(self.n_blocks):
                self.single_layers.append(
                    Op1QAllLayer_DST(op=tq.U3, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.control_layers.append(
                    Op2QAllLayer_DST(op=tq.CU3, n_wires=self.n_wires,
                                    has_params=True, trainable=True, circular=True))

        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
            self.q_device = q_device
            for k in range(self.n_blocks):
                if self.single_layers[k].op_flag.sum().cpu().item():
                    self.single_layers[k](self.q_device)
                if self.control_layers[k].op_flag.sum().cpu().item():
                    self.control_layers[k](self.q_device)

        def reset_sampling(self):
            for k in range(self.n_blocks):
                self.single_layers[k].reset_ops_sampling()
                self.control_layers[k].reset_ops_sampling()

    def __init__(self, n_wires=4, num_class=4, n_blocks=8):
        super().__init__()
        self.n_wires = n_wires
        self.num_class = num_class
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict['4x4_ryzxy'])

        self.q_layer = self.QLayer(self.n_wires, n_blocks)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x, use_qiskit=False):
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6).view(bsz, 16)

        if use_qiskit:
            x = self.qiskit_processor.process_parameterized(
                self.q_device, self.encoder, self.q_layer, self.measure, x)
        else:
            self.encoder(self.q_device, x)
            self.q_layer(self.q_device)
            x = self.measure(self.q_device)

        x = x.reshape(bsz, 4)

        return x

class ZZRY_Model_DST(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires, n_blocks):
            super().__init__()

            self.n_wires = n_wires
            self.n_blocks = n_blocks

            self.single_layers = tq.QuantumModuleList()
            self.control_layers = tq.QuantumModuleList()
            for _ in range(self.n_blocks):
                self.single_layers.append(
                    Op1QAllLayer_DST(op=tq.RY, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.control_layers.append(
                    Op2QAllLayer_DST(op=tq.RZZ, n_wires=self.n_wires,
                                    has_params=True, trainable=True, circular=True))
        
        # RZZ-Ring + RY
        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
            self.q_device = q_device
            for k in range(self.n_blocks):
                if self.control_layers[k].op_flag.sum().cpu().item():
                    self.control_layers[k](self.q_device)
                if self.single_layers[k].op_flag.sum().cpu().item():
                    self.single_layers[k](self.q_device)

        def reset_sampling(self):
            for k in range(self.n_blocks):
                self.control_layers[k].reset_ops_sampling()
                self.single_layers[k].reset_ops_sampling()

    def __init__(self, n_wires=4, num_class=4, n_blocks=8):
        super().__init__()
        self.n_wires = n_wires
        self.num_class = num_class
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict['4x4_ryzxy'])

        self.q_layer = self.QLayer(self.n_wires, n_blocks)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x, use_qiskit=False):
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6).view(bsz, 16)

        if use_qiskit:
            x = self.qiskit_processor.process_parameterized(
                self.q_device, self.encoder, self.q_layer, self.measure, x)
        else:
            self.encoder(self.q_device, x)
            self.q_layer(self.q_device)
            x = self.measure(self.q_device)

        x = x.reshape(bsz, 4)

        return x

class RXYZ_Model_DST(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires, n_blocks):
            super().__init__()

            self.n_wires = n_wires
            self.n_blocks = n_blocks
            
            self.hadamard = tq.Op1QAllLayer(op=tq.SHadamard, n_wires=self.n_wires)
            self.single_layers_1 = tq.QuantumModuleList()
            self.single_layers_2 = tq.QuantumModuleList()
            self.single_layers_3 = tq.QuantumModuleList()
            self.control_layers = tq.QuantumModuleList()
            for _ in range(self.n_blocks):
                self.single_layers_1.append(
                    Op1QAllLayer_DST(op=tq.RX, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.single_layers_2.append(
                    Op1QAllLayer_DST(op=tq.RY, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.single_layers_3.append(
                    Op1QAllLayer_DST(op=tq.RZ, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.control_layers.append(
                    tq.Op2QAllLayer(op=tq.CZ, n_wires=self.n_wires))
        
        # RX + RY + RZ + CZ
        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
            self.q_device = q_device
            self.hadamard(self.q_device)
            for k in range(self.n_blocks):
                if self.single_layers_1[k].op_flag.sum().cpu().item():
                    self.single_layers_1[k](self.q_device)
                if self.single_layers_2[k].op_flag.sum().cpu().item():
                    self.single_layers_2[k](self.q_device)
                if self.single_layers_3[k].op_flag.sum().cpu().item():
                    self.single_layers_3[k](self.q_device)
                self.control_layers[k](self.q_device)

        def reset_sampling(self):
            for k in range(self.n_blocks):
                self.single_layers_1[k].reset_ops_sampling()
                self.single_layers_2[k].reset_ops_sampling()
                self.single_layers_3[k].reset_ops_sampling()

    def __init__(self, n_wires=4, num_class=4, n_blocks=8):
        super().__init__()
        self.n_wires = n_wires
        self.num_class = num_class
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict['4x4_ryzxy'])

        self.q_layer = self.QLayer(self.n_wires, n_blocks)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x, use_qiskit=False):
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6).view(bsz, 16)

        if use_qiskit:
            x = self.qiskit_processor.process_parameterized(
                self.q_device, self.encoder, self.q_layer, self.measure, x)
        else:
            self.encoder(self.q_device, x)
            self.q_layer(self.q_device)
            x = self.measure(self.q_device)

        x = x.reshape(bsz, 4)

        return x

class ZXXX_Model_DST(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires, n_blocks):
            super().__init__()

            self.n_wires = n_wires
            self.n_blocks = n_blocks

            self.control_layers_1 = tq.QuantumModuleList()
            self.control_layers_2 = tq.QuantumModuleList()
            for _ in range(self.n_blocks):
                self.control_layers_1.append(
                    Op2QAllLayer_DST(op=tq.RZX, n_wires=self.n_wires,
                                    has_params=True, trainable=True, circular=True))
                self.control_layers_2.append(
                    Op2QAllLayer_DST(op=tq.RXX, n_wires=self.n_wires,
                                    has_params=True, trainable=True, circular=True))
        
        # RZX + RXX
        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
            self.q_device = q_device
            for k in range(self.n_blocks):
                if self.control_layers_1[k].op_flag.sum().cpu().item():
                    self.control_layers_1[k](self.q_device)
                if self.control_layers_2[k].op_flag.sum().cpu().item():
                    self.control_layers_2[k](self.q_device)

        def reset_sampling(self):
            for k in range(self.n_blocks):
                self.control_layers_1[k].reset_ops_sampling()
                self.control_layers_2[k].reset_ops_sampling()

    def __init__(self, n_wires=4, num_class=4, n_blocks=8):
        super().__init__()
        self.n_wires = n_wires
        self.num_class = num_class
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict['4x4_ryzxy'])

        self.q_layer = self.QLayer(self.n_wires, n_blocks)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x, use_qiskit=False):
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6).view(bsz, 16)

        if use_qiskit:
            x = self.qiskit_processor.process_parameterized(
                self.q_device, self.encoder, self.q_layer, self.measure, x)
        else:
            self.encoder(self.q_device, x)
            self.q_layer(self.q_device)
            x = self.measure(self.q_device)

        x = x.reshape(bsz, 4)

        return x

class RXYZ_U1_CU3_Model_DST(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires, n_blocks):
            super().__init__()

            self.n_wires = n_wires
            self.n_blocks = n_blocks

            self.single_layers_1 = tq.QuantumModuleList()
            self.single_layers_2 = tq.QuantumModuleList()
            self.single_layers_3 = tq.QuantumModuleList()
            self.single_layers_4 = tq.QuantumModuleList()
            self.single_layers_5 = tq.QuantumModuleList()
            self.single_layers_6 = tq.QuantumModuleList()
            self.single_layers_7 = tq.QuantumModuleList()

            self.control_layers_1 = tq.QuantumModuleList()
            self.control_layers_2 = tq.QuantumModuleList()
            self.control_layers_3 = tq.QuantumModuleList()
            self.control_layers_4 = tq.QuantumModuleList()

            for _ in range(self.n_blocks):
                self.single_layers_1.append(
                    Op1QAllLayer_DST(op=tq.RX, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.single_layers_2.append(
                    tq.Op1QAllLayer(op=tq.S, n_wires=self.n_wires))
                self.single_layers_3.append(
                    Op1QAllLayer_DST(op=tq.RY, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.single_layers_4.append(
                    tq.Op1QAllLayer(op=tq.T, n_wires=self.n_wires))
                self.single_layers_5.append(
                    Op1QAllLayer_DST(op=tq.RZ, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.single_layers_6.append(
                    tq.Op1QAllLayer(op=tq.T, n_wires=self.n_wires))
                self.single_layers_7.append(
                    Op1QAllLayer_DST(op=tq.U1, n_wires=self.n_wires,
                                    has_params=True, trainable=True))

                self.control_layers_1.append(
                    tq.Op2QAllLayer(op=tq.CNOT, n_wires=self.n_wires, circular=True))
                self.control_layers_2.append(
                    tq.Op2QAllLayer(op=tq.SWAP, n_wires=self.n_wires, circular=True))
                self.control_layers_3.append(
                    tq.Op2QAllLayer(op=tq.SSWAP, n_wires=self.n_wires, circular=True))
                self.control_layers_4.append(
                    Op2QAllLayer_DST(op=tq.CU3, n_wires=self.n_wires, 
                                    has_params=True, trainable=True, circular=True))

        # RXYZ + U1 + CU3
        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
            self.q_device = q_device
            for k in range(self.n_blocks):
                if self.single_layers_1[k].op_flag.sum().cpu().item():
                    self.single_layers_1[k](self.q_device)
                self.single_layers_2[k](self.q_device)
                self.control_layers_1[k](self.q_device)
                if self.single_layers_3[k].op_flag.sum().cpu().item():
                    self.single_layers_3[k](self.q_device)
                self.single_layers_4[k](self.q_device)
                self.control_layers_2[k](self.q_device)
                if self.single_layers_5[k].op_flag.sum().cpu().item():
                    self.single_layers_5[k](self.q_device)
                self.single_layers_6[k](self.q_device)
                self.control_layers_3[k](self.q_device)
                if self.single_layers_7[k].op_flag.sum().cpu().item():
                    self.single_layers_7[k](self.q_device)
                if self.control_layers_4[k].op_flag.sum().cpu().item():
                    self.control_layers_4[k](self.q_device)

        def reset_sampling(self):
            for k in range(self.n_blocks):
                self.single_layers_1[k].reset_ops_sampling()
                self.single_layers_3[k].reset_ops_sampling()
                self.single_layers_5[k].reset_ops_sampling()
                self.single_layers_7[k].reset_ops_sampling()
                self.control_layers_4[k].reset_ops_sampling()

    def __init__(self, n_wires=4, num_class=4, n_blocks=4):
        super().__init__()
        self.n_wires = n_wires
        self.num_class = num_class
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict['4x4_ryzxy'])

        self.q_layer = self.QLayer(self.n_wires, n_blocks)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x, use_qiskit=False):
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6).view(bsz, 16)

        if use_qiskit:
            x = self.qiskit_processor.process_parameterized(
                self.q_device, self.encoder, self.q_layer, self.measure, x)
        else:
            self.encoder(self.q_device, x)
            self.q_layer(self.q_device)
            x = self.measure(self.q_device)

        x = x.reshape(bsz, 4)

        return x

class IBMQ_Basis_Model_DST(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires, n_blocks):
            super().__init__()

            self.n_wires = n_wires
            self.n_blocks = n_blocks

            self.single_layers_1 = tq.QuantumModuleList()
            self.single_layers_2 = tq.QuantumModuleList()
            self.single_layers_3 = tq.QuantumModuleList()
            self.single_layers_4 = tq.QuantumModuleList()
            self.single_layers_5 = tq.QuantumModuleList()
            self.control_layers = tq.QuantumModuleList()

            for _ in range(self.n_blocks):
                self.single_layers_1.append(
                    Op1QAllLayer_DST(op=tq.RZ, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.single_layers_2.append(
                    tq.Op1QAllLayer(op=tq.PauliX, n_wires=self.n_wires))
                self.single_layers_3.append(
                    Op1QAllLayer_DST(op=tq.RZ, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.single_layers_4.append(
                    tq.Op1QAllLayer(op=tq.SX, n_wires=self.n_wires))
                self.single_layers_5.append(
                    Op1QAllLayer_DST(op=tq.RZ, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.control_layers.append(
                    tq.Op2QAllLayer(op=tq.CNOT, n_wires=self.n_wires, circular=True))

        # IBMQ-Basis
        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
            self.q_device = q_device
            for k in range(self.n_blocks):
                if self.single_layers_1[k].op_flag.sum().cpu().item():
                    self.single_layers_1[k](self.q_device)
                self.single_layers_2[k](self.q_device)
                if self.single_layers_3[k].op_flag.sum().cpu().item():
                    self.single_layers_3[k](self.q_device)
                self.single_layers_4[k](self.q_device)
                if self.single_layers_5[k].op_flag.sum().cpu().item():
                    self.single_layers_5[k](self.q_device)
                self.control_layers[k](self.q_device)

        def reset_sampling(self):
            for k in range(self.n_blocks):
                self.single_layers_1[k].reset_ops_sampling()
                self.single_layers_3[k].reset_ops_sampling()
                self.single_layers_5[k].reset_ops_sampling()

    def __init__(self, n_wires=4, num_class=4, n_blocks=20):
        super().__init__()
        self.n_wires = n_wires
        self.num_class = num_class
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict['4x4_ryzxy'])

        self.q_layer = self.QLayer(self.n_wires, n_blocks)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x, use_qiskit=False):
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6).view(bsz, 16)

        if use_qiskit:
            x = self.qiskit_processor.process_parameterized(
                self.q_device, self.encoder, self.q_layer, self.measure, x)
        else:
            self.encoder(self.q_device, x)
            self.q_layer(self.q_device)
            x = self.measure(self.q_device)

        x = x.reshape(bsz, 4)

        return x

class QFCModel_MNIST2_DST(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires=4, n_blocks=1):
            super().__init__()
            self.n_wires = n_wires
            self.blocks = n_blocks

            self.control_layers = tq.QuantumModuleList()
            self.single_layers = tq.QuantumModuleList()
            for k in range(self.blocks):
                self.control_layers.append(
                    Op2QAllLayer_DST(op=tq.RZZ, n_wires=self.n_wires,
                                    has_params=True, trainable=True, circular=True))
                self.single_layers.append(
                    Op1QAllLayer_DST(op=tq.RY, n_wires=self.n_wires,
                                    has_params=True, trainable=True))

        # 1 RZZ layer followed by 1 RY layer
        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
            self.q_device = q_device
            for k in range(self.blocks):
                if self.control_layers[k].op_flag.sum().cpu().item():
                    self.control_layers[k](self.q_device)
                if self.single_layers[k].op_flag.sum().cpu().item():
                    self.single_layers[k](self.q_device)

        def reset_sampling(self):
            for k in range(self.blocks):
                self.control_layers[k].reset_ops_sampling()
                self.single_layers[k].reset_ops_sampling()

    def __init__(self, n_wires=4, num_class=2, n_blocks=1):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict['4x4_ryzxy'])

        self.q_layer = self.QLayer(self.n_wires, n_blocks)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x, use_qiskit=False):
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6).view(bsz, 16)

        if use_qiskit:
            x = self.qiskit_processor.process_parameterized(
                self.q_device, self.encoder, self.q_layer, self.measure, x)
        else:
            self.encoder(self.q_device, x)
            self.q_layer(self.q_device)
            x = self.measure(self.q_device)

        x = x.reshape(bsz, 4)

        return x

class QFCModel_MNIST4_DST(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires=4, n_blocks=3):
            super().__init__()
            self.n_wires = n_wires
            self.blocks = n_blocks

            self.single_layers_1 = tq.QuantumModuleList()
            self.single_layers_2 = tq.QuantumModuleList()
            self.single_layers_3 = tq.QuantumModuleList()
            self.control_layers = tq.QuantumModuleList()

            for k in range(self.blocks):
                self.single_layers_1.append(
                    Op1QAllLayer_DST(op=tq.RX, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.single_layers_2.append(
                    Op1QAllLayer_DST(op=tq.RY, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.single_layers_3.append(
                    Op1QAllLayer_DST(op=tq.RZ, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.control_layers.append(
                    tq.Op2QAllLayer(op=tq.CZ, n_wires=self.n_wires,
                                    circular=True))

        # 3 RX+RY+RZ+CZ layers
        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
            self.q_device = q_device
            for k in range(self.blocks):
                if self.single_layers_1[k].op_flag.sum().cpu().item():
                    self.single_layers_1[k](self.q_device)
                if self.single_layers_2[k].op_flag.sum().cpu().item():
                    self.single_layers_2[k](self.q_device)
                if self.single_layers_3[k].op_flag.sum().cpu().item():
                    self.single_layers_3[k](self.q_device)
                self.control_layers[k](self.q_device)

        def reset_sampling(self):
            for k in range(self.blocks):
                self.single_layers_1[k].reset_ops_sampling()
                self.single_layers_2[k].reset_ops_sampling()
                self.single_layers_3[k].reset_ops_sampling()
                # skip control_layer which doesn't contain trainabla parameters

    def __init__(self, n_wires=4, num_class=4, n_blocks=3):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict['4x4_ryzxy'])

        self.q_layer = self.QLayer(self.n_wires, n_blocks)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x, use_qiskit=False):
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6).view(bsz, 16)

        if use_qiskit:
            x = self.qiskit_processor.process_parameterized(
                self.q_device, self.encoder, self.q_layer, self.measure, x)
        else:
            self.encoder(self.q_device, x)
            self.q_layer(self.q_device)
            x = self.measure(self.q_device)

        x = x.reshape(bsz, 4)

        return x



# vowel
class U3CU3_Model_DST_Vowel(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires, n_blocks):
            super().__init__()

            self.n_wires = n_wires
            self.n_blocks = n_blocks

            self.single_layers = tq.QuantumModuleList()
            self.control_layers = tq.QuantumModuleList()
            for _ in range(self.n_blocks):
                self.single_layers.append(
                    Op1QAllLayer_DST(op=tq.U3, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.control_layers.append(
                    Op2QAllLayer_DST(op=tq.CU3, n_wires=self.n_wires,
                                    has_params=True, trainable=True, circular=True))

        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
            self.q_device = q_device
            for k in range(self.n_blocks):
                if self.single_layers[k].op_flag.sum().cpu().item():
                    self.single_layers[k](self.q_device)
                if self.control_layers[k].op_flag.sum().cpu().item():
                    self.control_layers[k](self.q_device)

        def reset_sampling(self):
            for k in range(self.n_blocks):
                self.single_layers[k].reset_ops_sampling()
                self.control_layers[k].reset_ops_sampling()

    def __init__(self, n_wires=4, num_class=4, n_blocks=8):
        super().__init__()
        self.n_wires = n_wires
        self.num_class = num_class
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict['10_rxyz'])

        self.q_layer = self.QLayer(self.n_wires, n_blocks)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x, use_qiskit=False):
        bsz = x.shape[0]

        if use_qiskit:
            x = self.qiskit_processor.process_parameterized(
                self.q_device, self.encoder, self.q_layer, self.measure, x)
        else:
            self.encoder(self.q_device, x)
            self.q_layer(self.q_device)
            x = self.measure(self.q_device)

        x = x.reshape(bsz, 4)

        return x

class ZZRY_Model_DST_Vowel(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires, n_blocks):
            super().__init__()

            self.n_wires = n_wires
            self.n_blocks = n_blocks

            self.single_layers = tq.QuantumModuleList()
            self.control_layers = tq.QuantumModuleList()
            for _ in range(self.n_blocks):
                self.single_layers.append(
                    Op1QAllLayer_DST(op=tq.RY, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.control_layers.append(
                    Op2QAllLayer_DST(op=tq.RZZ, n_wires=self.n_wires,
                                    has_params=True, trainable=True, circular=True))
        
        # RZZ-Ring + RY
        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
            self.q_device = q_device
            for k in range(self.n_blocks):
                if self.control_layers[k].op_flag.sum().cpu().item():
                    self.control_layers[k](self.q_device)
                if self.single_layers[k].op_flag.sum().cpu().item():
                    self.single_layers[k](self.q_device)

        def reset_sampling(self):
            for k in range(self.n_blocks):
                self.control_layers[k].reset_ops_sampling()
                self.single_layers[k].reset_ops_sampling()

    def __init__(self, n_wires=4, num_class=4, n_blocks=8):
        super().__init__()
        self.n_wires = n_wires
        self.num_class = num_class
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict['10_rxyz'])

        self.q_layer = self.QLayer(self.n_wires, n_blocks)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x, use_qiskit=False):
        bsz = x.shape[0]

        if use_qiskit:
            x = self.qiskit_processor.process_parameterized(
                self.q_device, self.encoder, self.q_layer, self.measure, x)
        else:
            self.encoder(self.q_device, x)
            self.q_layer(self.q_device)
            x = self.measure(self.q_device)

        x = x.reshape(bsz, 4)

        return x

class RXYZ_Model_DST_Vowel(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires, n_blocks):
            super().__init__()

            self.n_wires = n_wires
            self.n_blocks = n_blocks
            
            self.hadamard = tq.Op1QAllLayer(op=tq.SHadamard, n_wires=self.n_wires)
            self.single_layers_1 = tq.QuantumModuleList()
            self.single_layers_2 = tq.QuantumModuleList()
            self.single_layers_3 = tq.QuantumModuleList()
            self.control_layers = tq.QuantumModuleList()
            for _ in range(self.n_blocks):
                self.single_layers_1.append(
                    Op1QAllLayer_DST(op=tq.RX, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.single_layers_2.append(
                    Op1QAllLayer_DST(op=tq.RY, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.single_layers_3.append(
                    Op1QAllLayer_DST(op=tq.RZ, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.control_layers.append(
                    tq.Op2QAllLayer(op=tq.CZ, n_wires=self.n_wires))
        
        # RX + RY + RZ + CZ
        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
            self.q_device = q_device
            self.hadamard(self.q_device)
            for k in range(self.n_blocks):
                if self.single_layers_1[k].op_flag.sum().cpu().item():
                    self.single_layers_1[k](self.q_device)
                if self.single_layers_2[k].op_flag.sum().cpu().item():
                    self.single_layers_2[k](self.q_device)
                if self.single_layers_3[k].op_flag.sum().cpu().item():
                    self.single_layers_3[k](self.q_device)
                self.control_layers[k](self.q_device)

        def reset_sampling(self):
            for k in range(self.n_blocks):
                self.single_layers_1[k].reset_ops_sampling()
                self.single_layers_2[k].reset_ops_sampling()
                self.single_layers_3[k].reset_ops_sampling()

    def __init__(self, n_wires=4, num_class=4, n_blocks=8):
        super().__init__()
        self.n_wires = n_wires
        self.num_class = num_class
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict['10_rxyz'])

        self.q_layer = self.QLayer(self.n_wires, n_blocks)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x, use_qiskit=False):
        bsz = x.shape[0]

        if use_qiskit:
            x = self.qiskit_processor.process_parameterized(
                self.q_device, self.encoder, self.q_layer, self.measure, x)
        else:
            self.encoder(self.q_device, x)
            self.q_layer(self.q_device)
            x = self.measure(self.q_device)

        x = x.reshape(bsz, 4)

        return x

class ZXXX_Model_DST_Vowel(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires, n_blocks):
            super().__init__()

            self.n_wires = n_wires
            self.n_blocks = n_blocks

            self.control_layers_1 = tq.QuantumModuleList()
            self.control_layers_2 = tq.QuantumModuleList()
            for _ in range(self.n_blocks):
                self.control_layers_1.append(
                    Op2QAllLayer_DST(op=tq.RZX, n_wires=self.n_wires,
                                    has_params=True, trainable=True, circular=True))
                self.control_layers_2.append(
                    Op2QAllLayer_DST(op=tq.RXX, n_wires=self.n_wires,
                                    has_params=True, trainable=True, circular=True))
        
        # RZX + RXX
        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
            self.q_device = q_device
            for k in range(self.n_blocks):
                if self.control_layers_1[k].op_flag.sum().cpu().item():
                    self.control_layers_1[k](self.q_device)
                if self.control_layers_2[k].op_flag.sum().cpu().item():
                    self.control_layers_2[k](self.q_device)

        def reset_sampling(self):
            for k in range(self.n_blocks):
                self.control_layers_1[k].reset_ops_sampling()
                self.control_layers_2[k].reset_ops_sampling()

    def __init__(self, n_wires=4, num_class=4, n_blocks=8):
        super().__init__()
        self.n_wires = n_wires
        self.num_class = num_class
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict['10_rxyz'])

        self.q_layer = self.QLayer(self.n_wires, n_blocks)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x, use_qiskit=False):
        bsz = x.shape[0]

        if use_qiskit:
            x = self.qiskit_processor.process_parameterized(
                self.q_device, self.encoder, self.q_layer, self.measure, x)
        else:
            self.encoder(self.q_device, x)
            self.q_layer(self.q_device)
            x = self.measure(self.q_device)

        x = x.reshape(bsz, 4)

        return x

class RXYZ_U1_CU3_Model_DST_Vowel(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires, n_blocks):
            super().__init__()

            self.n_wires = n_wires
            self.n_blocks = n_blocks

            self.single_layers_1 = tq.QuantumModuleList()
            self.single_layers_2 = tq.QuantumModuleList()
            self.single_layers_3 = tq.QuantumModuleList()
            self.single_layers_4 = tq.QuantumModuleList()
            self.single_layers_5 = tq.QuantumModuleList()
            self.single_layers_6 = tq.QuantumModuleList()
            self.single_layers_7 = tq.QuantumModuleList()

            self.control_layers_1 = tq.QuantumModuleList()
            self.control_layers_2 = tq.QuantumModuleList()
            self.control_layers_3 = tq.QuantumModuleList()
            self.control_layers_4 = tq.QuantumModuleList()

            for _ in range(self.n_blocks):
                self.single_layers_1.append(
                    Op1QAllLayer_DST(op=tq.RX, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.single_layers_2.append(
                    tq.Op1QAllLayer(op=tq.S, n_wires=self.n_wires))
                self.single_layers_3.append(
                    Op1QAllLayer_DST(op=tq.RY, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.single_layers_4.append(
                    tq.Op1QAllLayer(op=tq.T, n_wires=self.n_wires))
                self.single_layers_5.append(
                    Op1QAllLayer_DST(op=tq.RZ, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.single_layers_6.append(
                    tq.Op1QAllLayer(op=tq.T, n_wires=self.n_wires))
                self.single_layers_7.append(
                    Op1QAllLayer_DST(op=tq.U1, n_wires=self.n_wires,
                                    has_params=True, trainable=True))

                self.control_layers_1.append(
                    tq.Op2QAllLayer(op=tq.CNOT, n_wires=self.n_wires, circular=True))
                self.control_layers_2.append(
                    tq.Op2QAllLayer(op=tq.SWAP, n_wires=self.n_wires, circular=True))
                self.control_layers_3.append(
                    tq.Op2QAllLayer(op=tq.SSWAP, n_wires=self.n_wires, circular=True))
                self.control_layers_4.append(
                    Op2QAllLayer_DST(op=tq.CU3, n_wires=self.n_wires, 
                                    has_params=True, trainable=True, circular=True))

        # RXYZ + U1 + CU3
        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
            self.q_device = q_device
            for k in range(self.n_blocks):
                if self.single_layers_1[k].op_flag.sum().cpu().item():
                    self.single_layers_1[k](self.q_device)
                self.single_layers_2[k](self.q_device)
                self.control_layers_1[k](self.q_device)
                if self.single_layers_3[k].op_flag.sum().cpu().item():
                    self.single_layers_3[k](self.q_device)
                self.single_layers_4[k](self.q_device)
                self.control_layers_2[k](self.q_device)
                if self.single_layers_5[k].op_flag.sum().cpu().item():
                    self.single_layers_5[k](self.q_device)
                self.single_layers_6[k](self.q_device)
                self.control_layers_3[k](self.q_device)
                if self.single_layers_7[k].op_flag.sum().cpu().item():
                    self.single_layers_7[k](self.q_device)
                if self.control_layers_4[k].op_flag.sum().cpu().item():
                    self.control_layers_4[k](self.q_device)

        def reset_sampling(self):
            for k in range(self.n_blocks):
                self.single_layers_1[k].reset_ops_sampling()
                self.single_layers_3[k].reset_ops_sampling()
                self.single_layers_5[k].reset_ops_sampling()
                self.single_layers_7[k].reset_ops_sampling()
                self.control_layers_4[k].reset_ops_sampling()

    def __init__(self, n_wires=4, num_class=4, n_blocks=4):
        super().__init__()
        self.n_wires = n_wires
        self.num_class = num_class
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict['10_rxyz'])

        self.q_layer = self.QLayer(self.n_wires, n_blocks)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x, use_qiskit=False):
        bsz = x.shape[0]

        if use_qiskit:
            x = self.qiskit_processor.process_parameterized(
                self.q_device, self.encoder, self.q_layer, self.measure, x)
        else:
            self.encoder(self.q_device, x)
            self.q_layer(self.q_device)
            x = self.measure(self.q_device)

        x = x.reshape(bsz, 4)

        return x

class IBMQ_Basis_Model_DST_Vowel(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires, n_blocks):
            super().__init__()

            self.n_wires = n_wires
            self.n_blocks = n_blocks

            self.single_layers_1 = tq.QuantumModuleList()
            self.single_layers_2 = tq.QuantumModuleList()
            self.single_layers_3 = tq.QuantumModuleList()
            self.single_layers_4 = tq.QuantumModuleList()
            self.single_layers_5 = tq.QuantumModuleList()
            self.control_layers = tq.QuantumModuleList()

            for _ in range(self.n_blocks):
                self.single_layers_1.append(
                    Op1QAllLayer_DST(op=tq.RZ, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.single_layers_2.append(
                    tq.Op1QAllLayer(op=tq.PauliX, n_wires=self.n_wires))
                self.single_layers_3.append(
                    Op1QAllLayer_DST(op=tq.RZ, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.single_layers_4.append(
                    tq.Op1QAllLayer(op=tq.SX, n_wires=self.n_wires))
                self.single_layers_5.append(
                    Op1QAllLayer_DST(op=tq.RZ, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.control_layers.append(
                    tq.Op2QAllLayer(op=tq.CNOT, n_wires=self.n_wires, circular=True))

        # IBMQ-Basis
        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
            self.q_device = q_device
            for k in range(self.n_blocks):
                if self.single_layers_1[k].op_flag.sum().cpu().item():
                    self.single_layers_1[k](self.q_device)
                self.single_layers_2[k](self.q_device)
                if self.single_layers_3[k].op_flag.sum().cpu().item():
                    self.single_layers_3[k](self.q_device)
                self.single_layers_4[k](self.q_device)
                if self.single_layers_5[k].op_flag.sum().cpu().item():
                    self.single_layers_5[k](self.q_device)
                self.control_layers[k](self.q_device)

        def reset_sampling(self):
            for k in range(self.n_blocks):
                self.single_layers_1[k].reset_ops_sampling()
                self.single_layers_3[k].reset_ops_sampling()
                self.single_layers_5[k].reset_ops_sampling()

    def __init__(self, n_wires=4, num_class=4, n_blocks=20):
        super().__init__()
        self.n_wires = n_wires
        self.num_class = num_class
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict['10_rxyz'])

        self.q_layer = self.QLayer(self.n_wires, n_blocks)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x, use_qiskit=False):
        bsz = x.shape[0]

        if use_qiskit:
            x = self.qiskit_processor.process_parameterized(
                self.q_device, self.encoder, self.q_layer, self.measure, x)
        else:
            self.encoder(self.q_device, x)
            self.q_layer(self.q_device)
            x = self.measure(self.q_device)

        x = x.reshape(bsz, 4)

        return x




##  VQE ##
class U3CU3_Model_DST_VQE(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires, n_blocks):
            super().__init__()

            self.n_wires = n_wires
            self.n_blocks = n_blocks

            self.single_layers = tq.QuantumModuleList()
            self.control_layers = tq.QuantumModuleList()
            for _ in range(self.n_blocks):
                self.single_layers.append(
                    Op1QAllLayer_DST(op=tq.U3, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.control_layers.append(
                    Op2QAllLayer_DST(op=tq.CU3, n_wires=self.n_wires,
                                    has_params=True, trainable=True, circular=True))

        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
            self.q_device = q_device
            for k in range(self.n_blocks):
                if self.single_layers[k].op_flag.sum().cpu().item():
                    self.single_layers[k](self.q_device)
                if self.control_layers[k].op_flag.sum().cpu().item():
                    self.control_layers[k](self.q_device)

        def reset_sampling(self):
            for k in range(self.n_blocks):
                self.single_layers[k].reset_ops_sampling()
                self.control_layers[k].reset_ops_sampling()

    def __init__(self, hamil_info, n_wires=4, n_blocks=8):
        super().__init__()
        self.n_wires = n_wires
        self.hamil_info = hamil_info
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.q_layer = self.QLayer(self.n_wires, n_blocks)
        self.measure = tq.MeasureMultipleTimes(
            obs_list=hamil_info['hamil_list'])
        self.num_forwards = 0
        self.n_params = len(list(self.parameters()))

    def forward(self, x, verbose=False, use_qiskit=False):
        bsz = x.shape[0]

        if use_qiskit:
            x = self.qiskit_processor.process_multi_measure(
                self.q_device, self.q_layer, self.measure)
        else:
            self.q_device.reset_states(bsz=1)
            self.q_layer(self.q_device)
            x = self.measure(self.q_device)

        hamil_coefficients = torch.tensor([hamil['coefficient'] for hamil in
                                        self.hamil_info['hamil_list']],
                                        device=x.device).double()

        for k, hamil in enumerate(self.hamil_info['hamil_list']):
            for wire, observable in zip(hamil['wires'], hamil['observables']):
                if observable == 'i':
                    x[k][wire] = 1
            for wire in range(self.q_device.n_wires):
                if wire not in hamil['wires']:
                    x[k][wire] = 1

        if verbose:
            logger.info(f"[use_qiskit]={use_qiskit}, expectation:\n {x.data}")

        x = torch.cumprod(x, dim=-1)[:, -1].double()
        x = torch.dot(x, hamil_coefficients)

        if x.dim() == 0:
            x = x.unsqueeze(0)

        return x

    def shift_and_run(self, x, masker, verbose=False, use_qiskit=False, enable_skip=False):
        with torch.no_grad():
            if use_qiskit:
                self.q_device.reset_states(bsz=1)
                x = self.qiskit_processor.process_multi_measure(
                    self.q_device, self.q_layer, self.measure)
                self.grad_list = []

                for i, (name, param) in enumerate(self.named_parameters()):
                    if enable_skip:
                        param_valid = masker.masks[name + '_mask'].abs().sum() > 0
                        if not param_valid:
                            self.grad_list.append(0)
                            continue 
                    param.copy_(param + np.pi*0.5)
                    out1 = self.qiskit_processor.process_multi_measure(
                        self.q_device, self.q_layer, self.measure)
                    param.copy_(param - np.pi)
                    out2 = self.qiskit_processor.process_multi_measure(
                        self.q_device, self.q_layer, self.measure)
                    param.copy_(param + np.pi*0.5)
                    grad = 0.5 * (out1 - out2)
                    self.grad_list.append(grad)
            else:
                self.q_device.reset_states(bsz=1)
                self.q_layer(self.q_device)
                x = self.measure(self.q_device)
                self.grad_list = []
                for i, (name, param) in enumerate(self.named_parameters()):
                    if enable_skip:
                        param_valid = masker.masks[name + '_mask'].abs().sum() > 0
                        if not param_valid:
                            self.grad_list.append(0)
                            continue 

                    param.copy_(param + np.pi*0.5)
                    self.q_device.reset_states(bsz=1)
                    self.q_layer(self.q_device)
                    out1 = self.measure(self.q_device)
                    param.copy_(param - np.pi)
                    self.q_device.reset_states(bsz=1)
                    self.q_layer(self.q_device)
                    out2 = self.measure(self.q_device)
                    param.copy_(param + np.pi*0.5)
                    grad = 0.5 * (out1 - out2)
                    self.grad_list.append(grad)

        hamil_coefficients = torch.tensor([hamil['coefficient'] for hamil in
                                            self.hamil_info['hamil_list']],
                                            device=x.device).double()
        x_axis = []
        y_axis = []
        for k, hamil in enumerate(self.hamil_info['hamil_list']):
            for wire, observable in zip(hamil['wires'], hamil['observables']):
                if observable == 'i':
                    x[k][wire] = 1
                    x_axis.append(k)
                    y_axis.append(wire)
            for wire in range(self.q_device.n_wires):
                if wire not in hamil['wires']:
                    x[k][wire] = 1
                    x_axis.append(k)
                    y_axis.append(wire)
        for grad in self.grad_list:
            if not isinstance(grad, torch.Tensor): continue
            grad[x_axis, y_axis] = 0
        
        self.circuit_output = x
        self.circuit_output.requires_grad = True
        self.num_forwards += (1 + 2 * self.n_params) * x.shape[0]
        if verbose:
            logger.info(f"[use_qiskit]={use_qiskit}, expectation:\n {x.data}")

        x = torch.cumprod(x, dim=-1)[:, -1].double()
        x = torch.dot(x, hamil_coefficients)

        if x.dim() == 0:
            x = x.unsqueeze(0)

        return x

    def backprop_grad(self):
        for i, param in enumerate(self.parameters()):
            param.grad = torch.sum(self.grad_list[i] * self.circuit_output.grad).to(dtype=torch.float32).view(param.shape)

class ZZRY_Model_DST_VQE(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires, n_blocks):
            super().__init__()

            self.n_wires = n_wires
            self.n_blocks = n_blocks

            self.single_layers = tq.QuantumModuleList()
            self.control_layers = tq.QuantumModuleList()
            for _ in range(self.n_blocks):
                self.single_layers.append(
                    Op1QAllLayer_DST(op=tq.RY, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.control_layers.append(
                    Op2QAllLayer_DST(op=tq.RZZ, n_wires=self.n_wires,
                                    has_params=True, trainable=True, circular=True))
        
        # RZZ-Ring + RY
        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
            self.q_device = q_device
            for k in range(self.n_blocks):
                if self.control_layers[k].op_flag.sum().cpu().item():
                    self.control_layers[k](self.q_device)
                if self.single_layers[k].op_flag.sum().cpu().item():
                    self.single_layers[k](self.q_device)

        def reset_sampling(self):
            for k in range(self.n_blocks):
                self.control_layers[k].reset_ops_sampling()
                self.single_layers[k].reset_ops_sampling()

    def __init__(self, hamil_info, n_wires=4, num_class=4, n_blocks=8):
        super().__init__()
        self.n_wires = n_wires
        self.hamil_info = hamil_info
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.q_layer = self.QLayer(self.n_wires, n_blocks)
        self.measure = tq.MeasureMultipleTimes(
            obs_list=hamil_info['hamil_list'])
        self.num_forwards = 0
        self.n_params = len(list(self.parameters()))


    def forward(self, x, verbose=False, use_qiskit=False):
        bsz = x.shape[0]

        if use_qiskit:
            x = self.qiskit_processor.process_multi_measure(
                self.q_device, self.q_layer, self.measure)
        else:
            self.q_device.reset_states(bsz=1)
            self.q_layer(self.q_device)
            x = self.measure(self.q_device)

        hamil_coefficients = torch.tensor([hamil['coefficient'] for hamil in
                                        self.hamil_info['hamil_list']],
                                        device=x.device).double()

        for k, hamil in enumerate(self.hamil_info['hamil_list']):
            for wire, observable in zip(hamil['wires'], hamil['observables']):
                if observable == 'i':
                    x[k][wire] = 1
            for wire in range(self.q_device.n_wires):
                if wire not in hamil['wires']:
                    x[k][wire] = 1

        if verbose:
            logger.info(f"[use_qiskit]={use_qiskit}, expectation:\n {x.data}")

        x = torch.cumprod(x, dim=-1)[:, -1].double()
        x = torch.dot(x, hamil_coefficients)

        if x.dim() == 0:
            x = x.unsqueeze(0)

        return x

    def shift_and_run(self, x, masker, verbose=False, use_qiskit=False, enable_skip=False):
        with torch.no_grad():
            if use_qiskit:
                self.q_device.reset_states(bsz=1)
                x = self.qiskit_processor.process_multi_measure(
                    self.q_device, self.q_layer, self.measure)
                self.grad_list = []

                for i, (name, param) in enumerate(self.named_parameters()):
                    if enable_skip:
                        param_valid = masker.masks[name + '_mask'].abs().sum() > 0
                        if not param_valid:
                            self.grad_list.append(0)
                            continue 
                    param.copy_(param + np.pi*0.5)
                    out1 = self.qiskit_processor.process_multi_measure(
                        self.q_device, self.q_layer, self.measure)
                    param.copy_(param - np.pi)
                    out2 = self.qiskit_processor.process_multi_measure(
                        self.q_device, self.q_layer, self.measure)
                    param.copy_(param + np.pi*0.5)
                    grad = 0.5 * (out1 - out2)
                    self.grad_list.append(grad)
            else:
                self.q_device.reset_states(bsz=1)
                self.q_layer(self.q_device)
                x = self.measure(self.q_device)
                self.grad_list = []
                for i, (name, param) in enumerate(self.named_parameters()):
                    if enable_skip:
                        param_valid = masker.masks[name + '_mask'].abs().sum() > 0
                        if not param_valid:
                            self.grad_list.append(0)
                            continue 

                    param.copy_(param + np.pi*0.5)
                    self.q_device.reset_states(bsz=1)
                    self.q_layer(self.q_device)
                    out1 = self.measure(self.q_device)
                    param.copy_(param - np.pi)
                    self.q_device.reset_states(bsz=1)
                    self.q_layer(self.q_device)
                    out2 = self.measure(self.q_device)
                    param.copy_(param + np.pi*0.5)
                    grad = 0.5 * (out1 - out2)
                    self.grad_list.append(grad)

        hamil_coefficients = torch.tensor([hamil['coefficient'] for hamil in
                                            self.hamil_info['hamil_list']],
                                            device=x.device).double()
        x_axis = []
        y_axis = []
        for k, hamil in enumerate(self.hamil_info['hamil_list']):
            for wire, observable in zip(hamil['wires'], hamil['observables']):
                if observable == 'i':
                    x[k][wire] = 1
                    x_axis.append(k)
                    y_axis.append(wire)
            for wire in range(self.q_device.n_wires):
                if wire not in hamil['wires']:
                    x[k][wire] = 1
                    x_axis.append(k)
                    y_axis.append(wire)
        for grad in self.grad_list:
            if not isinstance(grad, torch.Tensor): continue
            grad[x_axis, y_axis] = 0
        
        self.circuit_output = x
        self.circuit_output.requires_grad = True
        self.num_forwards += (1 + 2 * self.n_params) * x.shape[0]
        if verbose:
            logger.info(f"[use_qiskit]={use_qiskit}, expectation:\n {x.data}")

        x = torch.cumprod(x, dim=-1)[:, -1].double()
        x = torch.dot(x, hamil_coefficients)

        if x.dim() == 0:
            x = x.unsqueeze(0)

        return x

    def backprop_grad(self):
        for i, param in enumerate(self.parameters()):
            param.grad = torch.sum(self.grad_list[i] * self.circuit_output.grad).to(dtype=torch.float32).view(param.shape)

class RXYZ_Model_DST_VQE(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires, n_blocks):
            super().__init__()

            self.n_wires = n_wires
            self.n_blocks = n_blocks
            
            self.hadamard = tq.Op1QAllLayer(op=tq.SHadamard, n_wires=self.n_wires)
            self.single_layers_1 = tq.QuantumModuleList()
            self.single_layers_2 = tq.QuantumModuleList()
            self.single_layers_3 = tq.QuantumModuleList()
            self.control_layers = tq.QuantumModuleList()
            for _ in range(self.n_blocks):
                self.single_layers_1.append(
                    Op1QAllLayer_DST(op=tq.RX, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.single_layers_2.append(
                    Op1QAllLayer_DST(op=tq.RY, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.single_layers_3.append(
                    Op1QAllLayer_DST(op=tq.RZ, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.control_layers.append(
                    tq.Op2QAllLayer(op=tq.CZ, n_wires=self.n_wires))
        
        # RX + RY + RZ + CZ
        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
            self.q_device = q_device
            self.hadamard(self.q_device)
            for k in range(self.n_blocks):
                if self.single_layers_1[k].op_flag.sum().cpu().item():
                    self.single_layers_1[k](self.q_device)
                if self.single_layers_2[k].op_flag.sum().cpu().item():
                    self.single_layers_2[k](self.q_device)
                if self.single_layers_3[k].op_flag.sum().cpu().item():
                    self.single_layers_3[k](self.q_device)
                self.control_layers[k](self.q_device)

        def reset_sampling(self):
            for k in range(self.n_blocks):
                self.single_layers_1[k].reset_ops_sampling()
                self.single_layers_2[k].reset_ops_sampling()
                self.single_layers_3[k].reset_ops_sampling()

    def __init__(self, hamil_info, n_wires=4, num_class=4, n_blocks=8):
        super().__init__()
        self.n_wires = n_wires
        self.hamil_info = hamil_info
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.q_layer = self.QLayer(self.n_wires, n_blocks)
        self.measure = tq.MeasureMultipleTimes(
            obs_list=hamil_info['hamil_list'])
        self.num_forwards = 0
        self.n_params = len(list(self.parameters()))


    def forward(self, x, verbose=False, use_qiskit=False):
        bsz = x.shape[0]

        if use_qiskit:
            x = self.qiskit_processor.process_multi_measure(
                self.q_device, self.q_layer, self.measure)
        else:
            self.q_device.reset_states(bsz=1)
            self.q_layer(self.q_device)
            x = self.measure(self.q_device)

        hamil_coefficients = torch.tensor([hamil['coefficient'] for hamil in
                                        self.hamil_info['hamil_list']],
                                        device=x.device).double()

        for k, hamil in enumerate(self.hamil_info['hamil_list']):
            for wire, observable in zip(hamil['wires'], hamil['observables']):
                if observable == 'i':
                    x[k][wire] = 1
            for wire in range(self.q_device.n_wires):
                if wire not in hamil['wires']:
                    x[k][wire] = 1

        if verbose:
            logger.info(f"[use_qiskit]={use_qiskit}, expectation:\n {x.data}")

        x = torch.cumprod(x, dim=-1)[:, -1].double()
        x = torch.dot(x, hamil_coefficients)

        if x.dim() == 0:
            x = x.unsqueeze(0)

        return x

    def shift_and_run(self, x, masker, verbose=False, use_qiskit=False, enable_skip=False):
        with torch.no_grad():
            if use_qiskit:
                self.q_device.reset_states(bsz=1)
                x = self.qiskit_processor.process_multi_measure(
                    self.q_device, self.q_layer, self.measure)
                self.grad_list = []

                for i, (name, param) in enumerate(self.named_parameters()):
                    if enable_skip:
                        param_valid = masker.masks[name + '_mask'].abs().sum() > 0
                        if not param_valid:
                            self.grad_list.append(0)
                            continue 
                    param.copy_(param + np.pi*0.5)
                    out1 = self.qiskit_processor.process_multi_measure(
                        self.q_device, self.q_layer, self.measure)
                    param.copy_(param - np.pi)
                    out2 = self.qiskit_processor.process_multi_measure(
                        self.q_device, self.q_layer, self.measure)
                    param.copy_(param + np.pi*0.5)
                    grad = 0.5 * (out1 - out2)
                    self.grad_list.append(grad)
            else:
                self.q_device.reset_states(bsz=1)
                self.q_layer(self.q_device)
                x = self.measure(self.q_device)
                self.grad_list = []
                for i, (name, param) in enumerate(self.named_parameters()):
                    if enable_skip:
                        param_valid = masker.masks[name + '_mask'].abs().sum() > 0
                        if not param_valid:
                            self.grad_list.append(0)
                            continue 

                    param.copy_(param + np.pi*0.5)
                    self.q_device.reset_states(bsz=1)
                    self.q_layer(self.q_device)
                    out1 = self.measure(self.q_device)
                    param.copy_(param - np.pi)
                    self.q_device.reset_states(bsz=1)
                    self.q_layer(self.q_device)
                    out2 = self.measure(self.q_device)
                    param.copy_(param + np.pi*0.5)
                    grad = 0.5 * (out1 - out2)
                    self.grad_list.append(grad)

        hamil_coefficients = torch.tensor([hamil['coefficient'] for hamil in
                                            self.hamil_info['hamil_list']],
                                            device=x.device).double()
        x_axis = []
        y_axis = []
        for k, hamil in enumerate(self.hamil_info['hamil_list']):
            for wire, observable in zip(hamil['wires'], hamil['observables']):
                if observable == 'i':
                    x[k][wire] = 1
                    x_axis.append(k)
                    y_axis.append(wire)
            for wire in range(self.q_device.n_wires):
                if wire not in hamil['wires']:
                    x[k][wire] = 1
                    x_axis.append(k)
                    y_axis.append(wire)
        for grad in self.grad_list:
            if not isinstance(grad, torch.Tensor): continue
            grad[x_axis, y_axis] = 0
        
        self.circuit_output = x
        self.circuit_output.requires_grad = True
        self.num_forwards += (1 + 2 * self.n_params) * x.shape[0]
        if verbose:
            logger.info(f"[use_qiskit]={use_qiskit}, expectation:\n {x.data}")

        x = torch.cumprod(x, dim=-1)[:, -1].double()
        x = torch.dot(x, hamil_coefficients)

        if x.dim() == 0:
            x = x.unsqueeze(0)

        return x

    def backprop_grad(self):
        for i, param in enumerate(self.parameters()):
            param.grad = torch.sum(self.grad_list[i] * self.circuit_output.grad).to(dtype=torch.float32).view(param.shape)

class ZXXX_Model_DST_VQE(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires, n_blocks):
            super().__init__()

            self.n_wires = n_wires
            self.n_blocks = n_blocks

            self.control_layers_1 = tq.QuantumModuleList()
            self.control_layers_2 = tq.QuantumModuleList()
            for _ in range(self.n_blocks):
                self.control_layers_1.append(
                    Op2QAllLayer_DST(op=tq.RZX, n_wires=self.n_wires,
                                    has_params=True, trainable=True, circular=True))
                self.control_layers_2.append(
                    Op2QAllLayer_DST(op=tq.RXX, n_wires=self.n_wires,
                                    has_params=True, trainable=True, circular=True))
        
        # RZX + RXX
        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
            self.q_device = q_device
            for k in range(self.n_blocks):
                if self.control_layers_1[k].op_flag.sum().cpu().item():
                    self.control_layers_1[k](self.q_device)
                if self.control_layers_2[k].op_flag.sum().cpu().item():
                    self.control_layers_2[k](self.q_device)

        def reset_sampling(self):
            for k in range(self.n_blocks):
                self.control_layers_1[k].reset_ops_sampling()
                self.control_layers_2[k].reset_ops_sampling()

    def __init__(self, hamil_info, n_wires=4, num_class=4, n_blocks=8):
        super().__init__()
        self.n_wires = n_wires
        self.hamil_info = hamil_info
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.q_layer = self.QLayer(self.n_wires, n_blocks)
        self.measure = tq.MeasureMultipleTimes(
            obs_list=hamil_info['hamil_list'])
        self.num_forwards = 0
        self.n_params = len(list(self.parameters()))


    def forward(self, x, verbose=False, use_qiskit=False):
        bsz = x.shape[0]

        if use_qiskit:
            x = self.qiskit_processor.process_multi_measure(
                self.q_device, self.q_layer, self.measure)
        else:
            self.q_device.reset_states(bsz=1)
            self.q_layer(self.q_device)
            x = self.measure(self.q_device)

        hamil_coefficients = torch.tensor([hamil['coefficient'] for hamil in
                                        self.hamil_info['hamil_list']],
                                        device=x.device).double()

        for k, hamil in enumerate(self.hamil_info['hamil_list']):
            for wire, observable in zip(hamil['wires'], hamil['observables']):
                if observable == 'i':
                    x[k][wire] = 1
            for wire in range(self.q_device.n_wires):
                if wire not in hamil['wires']:
                    x[k][wire] = 1

        if verbose:
            logger.info(f"[use_qiskit]={use_qiskit}, expectation:\n {x.data}")

        x = torch.cumprod(x, dim=-1)[:, -1].double()
        x = torch.dot(x, hamil_coefficients)

        if x.dim() == 0:
            x = x.unsqueeze(0)

        return x

    def shift_and_run(self, x, masker, verbose=False, use_qiskit=False, enable_skip=False):
        with torch.no_grad():
            if use_qiskit:
                self.q_device.reset_states(bsz=1)
                x = self.qiskit_processor.process_multi_measure(
                    self.q_device, self.q_layer, self.measure)
                self.grad_list = []

                for i, (name, param) in enumerate(self.named_parameters()):
                    if enable_skip:
                        param_valid = masker.masks[name + '_mask'].abs().sum() > 0
                        if not param_valid:
                            self.grad_list.append(0)
                            continue 
                    param.copy_(param + np.pi*0.5)
                    out1 = self.qiskit_processor.process_multi_measure(
                        self.q_device, self.q_layer, self.measure)
                    param.copy_(param - np.pi)
                    out2 = self.qiskit_processor.process_multi_measure(
                        self.q_device, self.q_layer, self.measure)
                    param.copy_(param + np.pi*0.5)
                    grad = 0.5 * (out1 - out2)
                    self.grad_list.append(grad)
            else:
                self.q_device.reset_states(bsz=1)
                self.q_layer(self.q_device)
                x = self.measure(self.q_device)
                self.grad_list = []
                for i, (name, param) in enumerate(self.named_parameters()):
                    if enable_skip:
                        param_valid = masker.masks[name + '_mask'].abs().sum() > 0
                        if not param_valid:
                            self.grad_list.append(0)
                            continue 

                    param.copy_(param + np.pi*0.5)
                    self.q_device.reset_states(bsz=1)
                    self.q_layer(self.q_device)
                    out1 = self.measure(self.q_device)
                    param.copy_(param - np.pi)
                    self.q_device.reset_states(bsz=1)
                    self.q_layer(self.q_device)
                    out2 = self.measure(self.q_device)
                    param.copy_(param + np.pi*0.5)
                    grad = 0.5 * (out1 - out2)
                    self.grad_list.append(grad)

        hamil_coefficients = torch.tensor([hamil['coefficient'] for hamil in
                                            self.hamil_info['hamil_list']],
                                            device=x.device).double()
        x_axis = []
        y_axis = []
        for k, hamil in enumerate(self.hamil_info['hamil_list']):
            for wire, observable in zip(hamil['wires'], hamil['observables']):
                if observable == 'i':
                    x[k][wire] = 1
                    x_axis.append(k)
                    y_axis.append(wire)
            for wire in range(self.q_device.n_wires):
                if wire not in hamil['wires']:
                    x[k][wire] = 1
                    x_axis.append(k)
                    y_axis.append(wire)
        for grad in self.grad_list:
            if not isinstance(grad, torch.Tensor): continue
            grad[x_axis, y_axis] = 0
        
        self.circuit_output = x
        self.circuit_output.requires_grad = True
        self.num_forwards += (1 + 2 * self.n_params) * x.shape[0]
        if verbose:
            logger.info(f"[use_qiskit]={use_qiskit}, expectation:\n {x.data}")

        x = torch.cumprod(x, dim=-1)[:, -1].double()
        x = torch.dot(x, hamil_coefficients)

        if x.dim() == 0:
            x = x.unsqueeze(0)

        return x
    
    def backprop_grad(self):
        for i, param in enumerate(self.parameters()):
            param.grad = torch.sum(self.grad_list[i] * self.circuit_output.grad).to(dtype=torch.float32).view(param.shape)

class RXYZ_U1_CU3_Model_DST_VQE(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires, n_blocks):
            super().__init__()

            self.n_wires = n_wires
            self.n_blocks = n_blocks

            self.single_layers_1 = tq.QuantumModuleList()
            self.single_layers_2 = tq.QuantumModuleList()
            self.single_layers_3 = tq.QuantumModuleList()
            self.single_layers_4 = tq.QuantumModuleList()
            self.single_layers_5 = tq.QuantumModuleList()
            self.single_layers_6 = tq.QuantumModuleList()
            self.single_layers_7 = tq.QuantumModuleList()

            self.control_layers_1 = tq.QuantumModuleList()
            self.control_layers_2 = tq.QuantumModuleList()
            self.control_layers_3 = tq.QuantumModuleList()
            self.control_layers_4 = tq.QuantumModuleList()

            for _ in range(self.n_blocks):
                self.single_layers_1.append(
                    Op1QAllLayer_DST(op=tq.RX, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.single_layers_2.append(
                    tq.Op1QAllLayer(op=tq.S, n_wires=self.n_wires))
                self.single_layers_3.append(
                    Op1QAllLayer_DST(op=tq.RY, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.single_layers_4.append(
                    tq.Op1QAllLayer(op=tq.T, n_wires=self.n_wires))
                self.single_layers_5.append(
                    Op1QAllLayer_DST(op=tq.RZ, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.single_layers_6.append(
                    tq.Op1QAllLayer(op=tq.T, n_wires=self.n_wires))
                self.single_layers_7.append(
                    Op1QAllLayer_DST(op=tq.U1, n_wires=self.n_wires,
                                    has_params=True, trainable=True))

                self.control_layers_1.append(
                    tq.Op2QAllLayer(op=tq.CNOT, n_wires=self.n_wires, circular=True))
                self.control_layers_2.append(
                    tq.Op2QAllLayer(op=tq.SWAP, n_wires=self.n_wires, circular=True))
                self.control_layers_3.append(
                    tq.Op2QAllLayer(op=tq.SSWAP, n_wires=self.n_wires, circular=True))
                self.control_layers_4.append(
                    Op2QAllLayer_DST(op=tq.CU3, n_wires=self.n_wires, 
                                    has_params=True, trainable=True, circular=True))

        # RXYZ + U1 + CU3
        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
            self.q_device = q_device
            for k in range(self.n_blocks):
                if self.single_layers_1[k].op_flag.sum().cpu().item():
                    self.single_layers_1[k](self.q_device)
                self.single_layers_2[k](self.q_device)
                self.control_layers_1[k](self.q_device)
                if self.single_layers_3[k].op_flag.sum().cpu().item():
                    self.single_layers_3[k](self.q_device)
                self.single_layers_4[k](self.q_device)
                self.control_layers_2[k](self.q_device)
                if self.single_layers_5[k].op_flag.sum().cpu().item():
                    self.single_layers_5[k](self.q_device)
                self.single_layers_6[k](self.q_device)
                self.control_layers_3[k](self.q_device)
                if self.single_layers_7[k].op_flag.sum().cpu().item():
                    self.single_layers_7[k](self.q_device)
                if self.control_layers_4[k].op_flag.sum().cpu().item():
                    self.control_layers_4[k](self.q_device)

        def reset_sampling(self):
            for k in range(self.n_blocks):
                self.single_layers_1[k].reset_ops_sampling()
                self.single_layers_3[k].reset_ops_sampling()
                self.single_layers_5[k].reset_ops_sampling()
                self.single_layers_7[k].reset_ops_sampling()
                self.control_layers_4[k].reset_ops_sampling()

    def __init__(self, hamil_info, n_wires=4, num_class=4, n_blocks=8):
        super().__init__()
        self.n_wires = n_wires
        self.hamil_info = hamil_info
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.q_layer = self.QLayer(self.n_wires, n_blocks)
        self.measure = tq.MeasureMultipleTimes(
            obs_list=hamil_info['hamil_list'])
        self.num_forwards = 0
        self.n_params = len(list(self.parameters()))



    def forward(self, x, verbose=False, use_qiskit=False):
        bsz = x.shape[0]

        if use_qiskit:
            x = self.qiskit_processor.process_multi_measure(
                self.q_device, self.q_layer, self.measure)
        else:
            self.q_device.reset_states(bsz=1)
            self.q_layer(self.q_device)
            x = self.measure(self.q_device)

        hamil_coefficients = torch.tensor([hamil['coefficient'] for hamil in
                                        self.hamil_info['hamil_list']],
                                        device=x.device).double()

        for k, hamil in enumerate(self.hamil_info['hamil_list']):
            for wire, observable in zip(hamil['wires'], hamil['observables']):
                if observable == 'i':
                    x[k][wire] = 1
            for wire in range(self.q_device.n_wires):
                if wire not in hamil['wires']:
                    x[k][wire] = 1

        if verbose:
            logger.info(f"[use_qiskit]={use_qiskit}, expectation:\n {x.data}")

        x = torch.cumprod(x, dim=-1)[:, -1].double()
        x = torch.dot(x, hamil_coefficients)

        if x.dim() == 0:
            x = x.unsqueeze(0)

        return x

    def shift_and_run(self, x, masker, verbose=False, use_qiskit=False, enable_skip=False):
        with torch.no_grad():
            if use_qiskit:
                self.q_device.reset_states(bsz=1)
                x = self.qiskit_processor.process_multi_measure(
                    self.q_device, self.q_layer, self.measure)
                self.grad_list = []

                for i, (name, param) in enumerate(self.named_parameters()):
                    if enable_skip:
                        param_valid = masker.masks[name + '_mask'].abs().sum() > 0
                        if not param_valid:
                            self.grad_list.append(0)
                            continue 
                    param.copy_(param + np.pi*0.5)
                    out1 = self.qiskit_processor.process_multi_measure(
                        self.q_device, self.q_layer, self.measure)
                    param.copy_(param - np.pi)
                    out2 = self.qiskit_processor.process_multi_measure(
                        self.q_device, self.q_layer, self.measure)
                    param.copy_(param + np.pi*0.5)
                    grad = 0.5 * (out1 - out2)
                    self.grad_list.append(grad)
            else:
                self.q_device.reset_states(bsz=1)
                self.q_layer(self.q_device)
                x = self.measure(self.q_device)
                self.grad_list = []
                for i, (name, param) in enumerate(self.named_parameters()):
                    if enable_skip:
                        param_valid = masker.masks[name + '_mask'].abs().sum() > 0
                        if not param_valid:
                            self.grad_list.append(0)
                            continue 

                    param.copy_(param + np.pi*0.5)
                    self.q_device.reset_states(bsz=1)
                    self.q_layer(self.q_device)
                    out1 = self.measure(self.q_device)
                    param.copy_(param - np.pi)
                    self.q_device.reset_states(bsz=1)
                    self.q_layer(self.q_device)
                    out2 = self.measure(self.q_device)
                    param.copy_(param + np.pi*0.5)
                    grad = 0.5 * (out1 - out2)
                    self.grad_list.append(grad)

        hamil_coefficients = torch.tensor([hamil['coefficient'] for hamil in
                                            self.hamil_info['hamil_list']],
                                            device=x.device).double()
        x_axis = []
        y_axis = []
        for k, hamil in enumerate(self.hamil_info['hamil_list']):
            for wire, observable in zip(hamil['wires'], hamil['observables']):
                if observable == 'i':
                    x[k][wire] = 1
                    x_axis.append(k)
                    y_axis.append(wire)
            for wire in range(self.q_device.n_wires):
                if wire not in hamil['wires']:
                    x[k][wire] = 1
                    x_axis.append(k)
                    y_axis.append(wire)
        for grad in self.grad_list:
            if not isinstance(grad, torch.Tensor): continue
            grad[x_axis, y_axis] = 0
        
        self.circuit_output = x
        self.circuit_output.requires_grad = True
        self.num_forwards += (1 + 2 * self.n_params) * x.shape[0]
        if verbose:
            logger.info(f"[use_qiskit]={use_qiskit}, expectation:\n {x.data}")

        x = torch.cumprod(x, dim=-1)[:, -1].double()
        x = torch.dot(x, hamil_coefficients)

        if x.dim() == 0:
            x = x.unsqueeze(0)

        return x

    def backprop_grad(self):
        for i, param in enumerate(self.parameters()):
            param.grad = torch.sum(self.grad_list[i] * self.circuit_output.grad).to(dtype=torch.float32).view(param.shape)

class IBMQ_Basis_Model_DST_VQE(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires, n_blocks):
            super().__init__()

            self.n_wires = n_wires
            self.n_blocks = n_blocks

            self.single_layers_1 = tq.QuantumModuleList()
            self.single_layers_2 = tq.QuantumModuleList()
            self.single_layers_3 = tq.QuantumModuleList()
            self.single_layers_4 = tq.QuantumModuleList()
            self.single_layers_5 = tq.QuantumModuleList()
            self.control_layers = tq.QuantumModuleList()

            for _ in range(self.n_blocks):
                self.single_layers_1.append(
                    Op1QAllLayer_DST(op=tq.RZ, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.single_layers_2.append(
                    tq.Op1QAllLayer(op=tq.PauliX, n_wires=self.n_wires))
                self.single_layers_3.append(
                    Op1QAllLayer_DST(op=tq.RZ, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.single_layers_4.append(
                    tq.Op1QAllLayer(op=tq.SX, n_wires=self.n_wires))
                self.single_layers_5.append(
                    Op1QAllLayer_DST(op=tq.RZ, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.control_layers.append(
                    tq.Op2QAllLayer(op=tq.CNOT, n_wires=self.n_wires, circular=True))

        # IBMQ-Basis
        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
            self.q_device = q_device
            for k in range(self.n_blocks):
                if self.single_layers_1[k].op_flag.sum().cpu().item():
                    self.single_layers_1[k](self.q_device)
                self.single_layers_2[k](self.q_device)
                if self.single_layers_3[k].op_flag.sum().cpu().item():
                    self.single_layers_3[k](self.q_device)
                self.single_layers_4[k](self.q_device)
                if self.single_layers_5[k].op_flag.sum().cpu().item():
                    self.single_layers_5[k](self.q_device)
                self.control_layers[k](self.q_device)

        def reset_sampling(self):
            for k in range(self.n_blocks):
                self.single_layers_1[k].reset_ops_sampling()
                self.single_layers_3[k].reset_ops_sampling()
                self.single_layers_5[k].reset_ops_sampling()

    def __init__(self, hamil_info, n_wires=4, num_class=4, n_blocks=8):
        super().__init__()
        self.n_wires = n_wires
        self.hamil_info = hamil_info
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.q_layer = self.QLayer(self.n_wires, n_blocks)
        self.measure = tq.MeasureMultipleTimes(
            obs_list=hamil_info['hamil_list'])
        self.num_forwards = 0
        self.n_params = len(list(self.parameters()))


    def forward(self, x, verbose=False, use_qiskit=False):
        bsz = x.shape[0]

        if use_qiskit:
            x = self.qiskit_processor.process_multi_measure(
                self.q_device, self.q_layer, self.measure)
        else:
            self.q_device.reset_states(bsz=1)
            self.q_layer(self.q_device)
            x = self.measure(self.q_device)

        hamil_coefficients = torch.tensor([hamil['coefficient'] for hamil in
                                        self.hamil_info['hamil_list']],
                                        device=x.device).double()

        for k, hamil in enumerate(self.hamil_info['hamil_list']):
            for wire, observable in zip(hamil['wires'], hamil['observables']):
                if observable == 'i':
                    x[k][wire] = 1
            for wire in range(self.q_device.n_wires):
                if wire not in hamil['wires']:
                    x[k][wire] = 1

        if verbose:
            logger.info(f"[use_qiskit]={use_qiskit}, expectation:\n {x.data}")

        x = torch.cumprod(x, dim=-1)[:, -1].double()
        x = torch.dot(x, hamil_coefficients)

        if x.dim() == 0:
            x = x.unsqueeze(0)

        return x

    def shift_and_run(self, x, masker, verbose=False, use_qiskit=False, enable_skip=False):
        with torch.no_grad():
            if use_qiskit:
                self.q_device.reset_states(bsz=1)
                x = self.qiskit_processor.process_multi_measure(
                    self.q_device, self.q_layer, self.measure)
                self.grad_list = []

                for i, (name, param) in enumerate(self.named_parameters()):
                    if enable_skip:
                        param_valid = masker.masks[name + '_mask'].abs().sum() > 0
                        if not param_valid:
                            self.grad_list.append(0)
                            continue 
                    param.copy_(param + np.pi*0.5)
                    out1 = self.qiskit_processor.process_multi_measure(
                        self.q_device, self.q_layer, self.measure)
                    param.copy_(param - np.pi)
                    out2 = self.qiskit_processor.process_multi_measure(
                        self.q_device, self.q_layer, self.measure)
                    param.copy_(param + np.pi*0.5)
                    grad = 0.5 * (out1 - out2)
                    self.grad_list.append(grad)
            else:
                self.q_device.reset_states(bsz=1)
                self.q_layer(self.q_device)
                x = self.measure(self.q_device)
                self.grad_list = []
                for i, (name, param) in enumerate(self.named_parameters()):
                    if enable_skip:
                        param_valid = masker.masks[name + '_mask'].abs().sum() > 0
                        if not param_valid:
                            self.grad_list.append(0)
                            continue 

                    param.copy_(param + np.pi*0.5)
                    self.q_device.reset_states(bsz=1)
                    self.q_layer(self.q_device)
                    out1 = self.measure(self.q_device)
                    param.copy_(param - np.pi)
                    self.q_device.reset_states(bsz=1)
                    self.q_layer(self.q_device)
                    out2 = self.measure(self.q_device)
                    param.copy_(param + np.pi*0.5)
                    grad = 0.5 * (out1 - out2)
                    self.grad_list.append(grad)

        hamil_coefficients = torch.tensor([hamil['coefficient'] for hamil in
                                            self.hamil_info['hamil_list']],
                                            device=x.device).double()
        x_axis = []
        y_axis = []
        for k, hamil in enumerate(self.hamil_info['hamil_list']):
            for wire, observable in zip(hamil['wires'], hamil['observables']):
                if observable == 'i':
                    x[k][wire] = 1
                    x_axis.append(k)
                    y_axis.append(wire)
            for wire in range(self.q_device.n_wires):
                if wire not in hamil['wires']:
                    x[k][wire] = 1
                    x_axis.append(k)
                    y_axis.append(wire)
        for grad in self.grad_list:
            if not isinstance(grad, torch.Tensor): continue
            grad[x_axis, y_axis] = 0
        
        self.circuit_output = x
        self.circuit_output.requires_grad = True
        self.num_forwards += (1 + 2 * self.n_params) * x.shape[0]
        if verbose:
            logger.info(f"[use_qiskit]={use_qiskit}, expectation:\n {x.data}")

        x = torch.cumprod(x, dim=-1)[:, -1].double()
        x = torch.dot(x, hamil_coefficients)

        if x.dim() == 0:
            x = x.unsqueeze(0)

        return x

    def backprop_grad(self):
        for i, param in enumerate(self.parameters()):
            param.grad = torch.sum(self.grad_list[i] * self.circuit_output.grad).to(dtype=torch.float32).view(param.shape)


class RXYZ_Model_DST_Quant(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires, n_blocks):
            super().__init__()

            self.n_wires = n_wires
            self.n_blocks = n_blocks
            
            self.hadamard = tq.Op1QAllLayer(op=tq.SHadamard, n_wires=self.n_wires)
            self.single_layers_1 = tq.QuantumModuleList()
            self.single_layers_2 = tq.QuantumModuleList()
            self.single_layers_3 = tq.QuantumModuleList()
            self.control_layers = tq.QuantumModuleList()
            for _ in range(self.n_blocks):
                self.single_layers_1.append(
                    Op1QAllLayer_DST(op=tq.RX, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.single_layers_2.append(
                    Op1QAllLayer_DST(op=tq.RY, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.single_layers_3.append(
                    Op1QAllLayer_DST(op=tq.RZ, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.control_layers.append(
                    tq.Op2QAllLayer(op=tq.CZ, n_wires=self.n_wires))
        
        # RX + RY + RZ + CZ
        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
            self.q_device = q_device
            self.hadamard(self.q_device)
            for k in range(self.n_blocks):
                if self.single_layers_1[k].op_flag.sum().cpu().item():
                    self.single_layers_1[k](self.q_device)
                if self.single_layers_2[k].op_flag.sum().cpu().item():
                    self.single_layers_2[k](self.q_device)
                if self.single_layers_3[k].op_flag.sum().cpu().item():
                    self.single_layers_3[k](self.q_device)
                self.control_layers[k](self.q_device)

        def reset_sampling(self):
            for k in range(self.n_blocks):
                self.single_layers_1[k].reset_ops_sampling()
                self.single_layers_2[k].reset_ops_sampling()
                self.single_layers_3[k].reset_ops_sampling()

    def __init__(self, n_wires=4, num_class=4, n_blocks=8):
        super().__init__()
        self.n_wires = n_wires
        self.num_class = num_class
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict['4x4_ryzxy'])

        self.q_layer = self.QLayer(self.n_wires, n_blocks)
        self.measure = tq.MeasureAll(tq.PauliZ)

        self.x_before_act_quant = None
        self.circuit_out = None
        self.grad_list = []
        self.bn = torch.nn.BatchNorm1d(
                num_features=n_wires,
                momentum=None,
                affine=False,
                track_running_stats=False
            )

    def run_circuit(self, x, use_qiskit=False):
        bsz = x.shape[0]
        if use_qiskit:
            x = self.qiskit_processor.process_parameterized(
                self.q_device, self.encoder, self.q_layer, self.measure, x)
        else:
            self.encoder(self.q_device, x)
            self.q_layer(self.q_device)
            x = self.measure(self.q_device)

        x = x.reshape(bsz, 4)
        return x

    def shift_and_run(self, inputs, masker, use_qiskit=False, enable_skip=False):
        param_list = []
        name_list = []
        for name, param in self.named_parameters():
            param_list.append(param)
            name_list.append(name + '_mask')
        self.grad_list = []
        for idx, param in enumerate(param_list):
            if enable_skip:
                with torch.no_grad():
                    param_valid = masker.masks[name_list[idx]].abs().sum() > 0
                if not param_valid:
                    self.grad_list.append(0)
                    continue 

            param.copy_(param + np.pi * 0.5)
            out1 = self.run_circuit(inputs, use_qiskit)
            param.copy_(param - np.pi)
            out2 = self.run_circuit(inputs, use_qiskit)
            param.copy_(param + np.pi * 0.5)
            grad = 0.5 * (out1 - out2)
            self.grad_list.append(grad)
        return self.run_circuit(inputs, use_qiskit)

    def forward(self, x, masker, use_qiskit=False, enable_skip=False, normalize=False):
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6).view(bsz, 16)

        with torch.no_grad():
            x = self.shift_and_run(x, masker, use_qiskit, enable_skip)

        self.circuit_out = x
        self.circuit_out.requires_grad = True

        if normalize:
            x = self.bn(x)
        self.x_before_act_quant = x.clone()

        return x

    def backprop_grad(self):
        for i, param in enumerate(self.parameters()):
            param.grad = torch.sum(self.grad_list[i] * self.circuit_out.grad).to(dtype=torch.float32).view(param.shape)


class RXYZ_Model_DST_Quant_GPU(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires, n_blocks):
            super().__init__()

            self.n_wires = n_wires
            self.n_blocks = n_blocks
            
            self.hadamard = tq.Op1QAllLayer(op=tq.SHadamard, n_wires=self.n_wires)
            self.single_layers_1 = tq.QuantumModuleList()
            self.single_layers_2 = tq.QuantumModuleList()
            self.single_layers_3 = tq.QuantumModuleList()
            self.control_layers = tq.QuantumModuleList()
            for _ in range(self.n_blocks):
                self.single_layers_1.append(
                    Op1QAllLayer_DST(op=tq.RX, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.single_layers_2.append(
                    Op1QAllLayer_DST(op=tq.RY, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.single_layers_3.append(
                    Op1QAllLayer_DST(op=tq.RZ, n_wires=self.n_wires,
                                    has_params=True, trainable=True))
                self.control_layers.append(
                    tq.Op2QAllLayer(op=tq.CZ, n_wires=self.n_wires))
        
        # RX + RY + RZ + CZ
        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
            self.q_device = q_device
            self.hadamard(self.q_device)
            for k in range(self.n_blocks):
                if self.single_layers_1[k].op_flag.sum().cpu().item():
                    self.single_layers_1[k](self.q_device)
                if self.single_layers_2[k].op_flag.sum().cpu().item():
                    self.single_layers_2[k](self.q_device)
                if self.single_layers_3[k].op_flag.sum().cpu().item():
                    self.single_layers_3[k](self.q_device)
                self.control_layers[k](self.q_device)

        def reset_sampling(self):
            for k in range(self.n_blocks):
                self.single_layers_1[k].reset_ops_sampling()
                self.single_layers_2[k].reset_ops_sampling()
                self.single_layers_3[k].reset_ops_sampling()

    def __init__(self, n_wires=4, num_class=4, n_blocks=8):
        super().__init__()
        self.n_wires = n_wires
        self.num_class = num_class
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict['4x4_ryzxy'])

        self.q_layer = self.QLayer(self.n_wires, n_blocks)
        self.measure = tq.MeasureAll(tq.PauliZ)

        self.x_before_act_quant = None
        self.bn = torch.nn.BatchNorm1d(
                num_features=n_wires,
                momentum=None,
                affine=False,
                track_running_stats=False
            )

    def forward(self, x, use_qiskit=False, normalize=False):
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6).view(bsz, 16)

        if use_qiskit:
            x = self.qiskit_processor.process_parameterized(
                self.q_device, self.encoder, self.q_layer, self.measure, x)
        else:
            self.encoder(self.q_device, x)
            self.q_layer(self.q_device)
            x = self.measure(self.q_device)

        x = x.reshape(bsz, 4)

        if normalize:
            x = self.bn(x)
        self.x_before_act_quant = x.clone()

        return x



############## Data ############

def get_dataflow(args):

    if args.dataset == 'mnist' and args.num_class == 2:
        print('Datset = MNIST-2')
        dataset = MNIST(
            root='./mnist_data',
            train_valid_split_ratio=[0.95, 0.05],
            center_crop=24,
            resize=28,
            resize_mode='bilinear',
            binarize=False,
            digits_of_interest=[3, 6],
            n_test_samples=args.n_test_images,
            n_valid_samples=args.n_val_images,
            n_train_samples=args.n_train_images
        )
    elif args.dataset == 'mnist' and args.num_class == 4:
        print('Datset = MNIST-4')
        dataset = MNIST(
            root='./mnist_data',
            train_valid_split_ratio=[0.95, 0.05],
            center_crop=24,
            resize=28,
            resize_mode='bilinear',
            binarize=False,
            digits_of_interest=[0,1,2,3],
            n_test_samples=args.n_test_images,
            n_valid_samples=args.n_val_images,
            n_train_samples=args.n_train_images
        )
    elif args.dataset == 'mnist' and args.num_class == 10:
        print('Datset = MNIST-10')
        dataset = MNIST(
            root='./mnist_data',
            train_valid_split_ratio=[0.95, 0.05],
            center_crop=24,
            resize=28,
            resize_mode='bilinear',
            binarize=False,
            n_test_samples=args.n_test_images,
            n_valid_samples=args.n_val_images,
            n_train_samples=args.n_train_images
        )
    elif args.dataset == 'fmnist' and args.num_class == 2:
        print('Datset = Fashion-MNIST-2')
        dataset = MNIST(
            root='./fashion_mnist_data',
            train_valid_split_ratio=[0.95, 0.05],
            center_crop=24,
            resize=28,
            resize_mode='bilinear',
            binarize=False,
            digits_of_interest=[3, 6],
            n_test_samples=args.n_test_images,
            n_valid_samples=args.n_val_images,
            n_train_samples=args.n_train_images,
            fashion=True
        )
    elif args.dataset == 'fmnist' and args.num_class == 4:
        print('Datset = Fashion-MNIST-4')
        dataset = MNIST(
            root='./fashion_mnist_data',
            train_valid_split_ratio=[0.95, 0.05],
            center_crop=24,
            resize=28,
            resize_mode='bilinear',
            binarize=False,
            digits_of_interest=[0,1,2,3],
            n_test_samples=args.n_test_images,
            n_valid_samples=args.n_val_images,
            n_train_samples=args.n_train_images,
            fashion=True
        )
    elif args.dataset == 'vowel' and args.num_class == 4:
        print('Datset = Vowel-4')
        dataset = Vowel(
            root='./vowel_data',
            test_ratio=0.3,
            train_valid_split_ratio=[0.85, 0.15], # 6:1:3
            resize=10,
            binarize=0,
            binarize_threshold=0,
            digits_of_interest=[0,5,1,6]
        )
    elif args.dataset == 'VQE':
        print('Dataet = VQE')
        dataset = VQE(
            steps_per_epoch=50
        )
    else:
        assert False

    dataflow = dict()
    for split in dataset:
        sampler = torch.utils.data.RandomSampler(dataset[split])
        dataflow[split] = torch.utils.data.DataLoader(
            dataset[split],
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=2,
            pin_memory=True)

    return dataflow


def shift_and_run(model, inputs, masker, use_qiskit=False, enable_skip=False, apply_sign=False):
    param_list = []
    name_list = []
    for name, param in model.named_parameters():
        param_list.append(param)
        name_list.append(name + '_mask')
    grad_list = []
    for idx, param in enumerate(param_list):
        if enable_skip:
            with torch.no_grad():
                param_valid = masker.masks[name_list[idx]].abs().sum() > 0
            if not param_valid:
                grad_list.append(0)
                continue 

        param.copy_(param + np.pi * 0.5)
        out1 = model(inputs, use_qiskit)
        param.copy_(param - np.pi)
        out2 = model(inputs, use_qiskit)
        param.copy_(param + np.pi * 0.5)
        grad = 0.5 * (out1 - out2)
        if apply_sign:
            grad = grad.sign()
        grad_list.append(grad)
    return model(inputs, use_qiskit), grad_list


def train_and_return_grad(dataflow, model, device, optimizer, args, masker, gradient_beta=0.9, skip_ops=False, qiskit=False):

    accumulate_iteration = 0
    accumulate_time = 0

    for feed_dict in dataflow['train']:

        start = time.time()
        inputs = feed_dict[args.input_name].to(device)
        targets = feed_dict[args.target_name].to(device)

        # calculate gradients via parameters shift rules
        with torch.no_grad():
            outputs, grad_list = shift_and_run(model, inputs, masker, use_qiskit=qiskit, enable_skip=skip_ops, apply_sign=args.apply_sign)

        outputs.requires_grad=True
        if args.num_class == 2:
            prediction = outputs.reshape(-1, 2, 2).sum(-1).squeeze()
        else:
            prediction = outputs
        loss = F.nll_loss(F.log_softmax(prediction, dim=1), targets)
        optimizer.zero_grad()
        loss.backward()
        for i, param in enumerate(model.q_layer.parameters()):
            param.grad = torch.sum(grad_list[i] * outputs.grad).to(dtype=torch.float32, device=param.device).view(param.shape)
        masker.step(gradient_beta)

        accumulate_time += time.time() - start
        accumulate_iteration += 1
        mean_time = accumulate_time / accumulate_iteration
        print(f"[{accumulate_iteration}/{len(dataflow['train'])}], loss: {loss.item()}, mean time = {mean_time}", end='\r')


def train_and_return_grad_quant(dataflow, model, device, optimizer, args, masker, gradient_beta=0.9, skip_ops=False, qiskit=False):

    accumulate_iteration = 0
    accumulate_time = 0

    for feed_dict in dataflow['train']:

        start = time.time()
        inputs = feed_dict[args.input_name].to(device)
        targets = feed_dict[args.target_name].to(device)

        outputs = model(inputs, masker, use_qiskit=qiskit, enable_skip=skip_ops, normalize=args.normalization)
        loss_mse = F.mse_loss(outputs, model.x_before_act_quant)

        if args.num_class == 2:
            prediction = outputs.reshape(-1, 2, 2).sum(-1).squeeze()
        else:
            prediction = outputs
        loss = F.nll_loss(F.log_softmax(prediction, dim=1), targets)
        loss += args.act_quant_loss_lambda * loss_mse
        optimizer.zero_grad()
        loss.backward()
        model.backprop_grad()
        masker.step(gradient_beta)

        accumulate_time += time.time() - start
        accumulate_iteration += 1
        mean_time = accumulate_time / accumulate_iteration
        print(f"[{accumulate_iteration}/{len(dataflow['train'])}], loss: {loss.item()}, mean time = {mean_time}", end='\r')




def train_vqe(dataflow, model, device, optimizer, masker, parameter_shift=True, gradient_beta=0.9, skip_ops=False, qiskit=False):

    accumulate_iteration = 0
    accumulate_time = 0

    for feed_dict in dataflow['train']:

        start = time.time()
        inputs = feed_dict['input'].to(device)
        targets = feed_dict['target'].to(device)

        if parameter_shift:
            outputs = model.shift_and_run(inputs, masker, enable_skip=skip_ops, verbose=False, use_qiskit=qiskit)
        else:
            outputs = model(inputs, verbose=False, use_qiskit=qiskit)

        loss = minimize(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        model.backprop_grad()

        masker.step(gradient_beta)

        accumulate_time += time.time() - start
        accumulate_iteration += 1
        mean_time = accumulate_time / accumulate_iteration
        print(f"[{accumulate_iteration}/{len(dataflow['train'])}], loss: {loss.item()}, mean time = {mean_time}", end='\r')

def test_vqe(dataflow, split, model, device, qiskit=False):
    loss_all = []
    with torch.no_grad():
        for feed_dict in dataflow[split]:
            inputs = feed_dict['input'].to(device)
            targets = feed_dict['target'].to(device)

            outputs = model(inputs, verbose=False, use_qiskit=qiskit)
            loss = minimize(outputs, targets)
            loss_all.append(loss.reshape(-1,1))

        loss_all = torch.cat(loss_all, dim=0)

    size = loss_all.shape[0]
    error = loss_all.sum().item() / size
    print(f"{split} set error qiskit:{qiskit}: {error}")
    return error


def train(dataflow, model, device, optimizer, args, masker, gradient_beta=0.9, qiskit=False):

    accumulate_iteration = 0
    accumulate_time = 0

    for feed_dict in dataflow['train']:

        start = time.time()
        inputs = feed_dict['image'].to(device)
        targets = feed_dict['digit'].to(device)

        outputs = model(inputs, use_qiskit=qiskit)
        if args.num_class == 2:
            outputs = outputs.reshape(-1, 2, 2).sum(-1).squeeze()
        else:
            outputs = outputs

        loss = F.nll_loss(F.log_softmax(outputs, dim=1), targets)
        optimizer.zero_grad()
        loss.backward()
        masker.step(gradient_beta)

        accumulate_time += time.time() - start
        accumulate_iteration += 1
        mean_time = accumulate_time / accumulate_iteration
        print(f"[{accumulate_iteration}/{len(dataflow['train'])}], loss: {loss.item()}, mean time = {mean_time}", end='\r')


def train_quant(dataflow, model, device, optimizer, args, masker, gradient_beta=0.9, qiskit=False):
    print('Valid for Stage2')

    accumulate_iteration = 0
    accumulate_time = 0

    for feed_dict in dataflow['train']:

        start = time.time()
        inputs = feed_dict['image'].to(device)
        targets = feed_dict['digit'].to(device)

        outputs = model(inputs, use_qiskit=qiskit, normalize=args.normalization)

        if args.quant:
            loss_mse = F.mse_loss(outputs, model.x_before_act_quant)
        else:
            loss_mse = 0

        if args.num_class == 2:
            outputs = outputs.reshape(-1, 2, 2).sum(-1).squeeze()
        else:
            outputs = outputs

        loss = F.nll_loss(F.log_softmax(outputs, dim=1), targets)
        loss += args.act_quant_loss_lambda * loss_mse

        optimizer.zero_grad()
        loss.backward()

        # set zero for None gradient
        with torch.no_grad():
            for p in model.parameters():
                if not p.grad:
                    p.grad = torch.zeros_like(p.data)

        masker.step(gradient_beta)

        accumulate_time += time.time() - start
        accumulate_iteration += 1
        mean_time = accumulate_time / accumulate_iteration
        print(f"[{accumulate_iteration}/{len(dataflow['train'])}], loss: {loss.item()}, mean time = {mean_time}", end='\r')




def valid_test(dataflow, split, model, device, qiskit=False):
    target_all = []
    output_all = []
    with torch.no_grad():
        for feed_dict in dataflow[split]:
            inputs = feed_dict['image'].to(device)
            targets = feed_dict['digit'].to(device)

            outputs = model(inputs, use_qiskit=qiskit)

            target_all.append(targets)
            output_all.append(outputs)
        target_all = torch.cat(target_all, dim=0)
        output_all = torch.cat(output_all, dim=0)

    _, indices = output_all.topk(1, dim=1)
    masks = indices.eq(target_all.view(-1, 1).expand_as(indices))
    size = target_all.shape[0]
    corrects = masks.sum().item()
    accuracy = corrects / size
    loss = F.nll_loss(output_all, target_all).item()

    print(f"{split} set accuracy: {accuracy}")
    print(f"{split} set loss: {loss}")
    return accuracy

def valid_test_v2(dataflow, split, model, device, args, qiskit=False):
    target_all = []
    output_all = []
    with torch.no_grad():
        for feed_dict in dataflow[split]:
            inputs = feed_dict[args.input_name].to(device)
            targets = feed_dict[args.target_name].to(device)

            outputs = model(inputs, use_qiskit=qiskit)
            if args.num_class == 2:
                prediction = outputs.reshape(-1, 2, 2).sum(-1).squeeze()
            else:
                prediction = outputs
            prediction = F.log_softmax(prediction, dim=1)

            target_all.append(targets)
            output_all.append(prediction)
        target_all = torch.cat(target_all, dim=0)
        output_all = torch.cat(output_all, dim=0)

    _, indices = output_all.topk(1, dim=1)
    masks = indices.eq(target_all.view(-1, 1).expand_as(indices))
    size = target_all.shape[0]
    corrects = masks.sum().item()
    accuracy = corrects / size
    loss = F.nll_loss(output_all, target_all).item()

    print(f"{split} set accuracy qiskit:{qiskit}: {accuracy}")
    print(f"{split} set loss qiskit:{qiskit}: {loss}")
    return accuracy






def growth_rand_beta(cur_epoch, num_epoch):
    return max(1 - cur_epoch / num_epoch, 0)



def growth_rand_beta_cosine(cur_epoch, num_epoch):
    return max(np.cos(np.pi * cur_epoch / (2 * num_epoch)), 0)





def check_sparsity(model):
    zero_num = 0
    all_num = 0
    for p in model.parameters():
        zero_num += p.eq(0).float().sum()
        all_num += p.nelement()
    print('Remain Parameters = [{}/{} = {:.2f}%]'.format(all_num - zero_num, all_num, 100*(all_num - zero_num)/all_num))


def setup_seed(seed): 
    print('setup random seed = {}'.format(seed))
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed) 
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True 


def complex_mse(output, target):
    return F.mse_loss(torch.view_as_real(output),
                      torch.view_as_real(target))


def complex_mae(output, target):
    return (torch.view_as_real(output)
            - torch.view_as_real(target)).abs().mean()


def minimize(output, target):
    return output.sum()


def maximize(output, target):
    return -output.sum()





