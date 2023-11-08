import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np

from enum import IntEnum
from torchquantum.functional import mat_dict
from abc import ABCMeta
from .macro import C_DTYPE, F_DTYPE
from torchpack.utils.logging import logger
from typing import Iterable

__all__ = [
    'op_name_dict',
    'Operator',
    'Operation',
    'DiagonalOperation',
    'Observable',
    'Hadamard',
    'SHadamard',
    'PauliX',
    'PauliY',
    'PauliZ',
    'I',
    'S',
    'T',
    'SX',
    'CNOT',
    'CZ',
    'CY',
    'RX',
    'RY',
    'RZ',
    'RXX',
    'RYY',
    'RZZ',
    'RZX',
    'SWAP',
    'SSWAP',
    'CSWAP',
    'Toffoli',
    'PhaseShift',
    'Rot',
    'MultiRZ',
    'CRX',
    'CRY',
    'CRZ',
    'CRot',
    'U1',
    'U2',
    'U3',
    'CU1',
    'CU2',
    'CU3',
    'QubitUnitary',
    'QubitUnitaryFast',
    'TrainableUnitary',
    'TrainableUnitaryStrict',
    'MultiCNOT',
    'MultiXCNOT',
    'Reset',
]


class WiresEnum(IntEnum):
    """Integer enumeration class
    to represent the number of wires
    an operation acts on"""
    AnyWires = -1
    AllWires = 0


class NParamsEnum(IntEnum):
    """Integer enumeration class
    to represent the number of wires
    an operation acts on"""
    AnyNParams = -1


AnyNParams = NParamsEnum.AnyNParams


AllWires = WiresEnum.AllWires
"""IntEnum: An enumeration which represents all wires in the
subsystem. It is equivalent to an integer with value 0."""

AnyWires = WiresEnum.AnyWires
"""IntEnum: An enumeration which represents any wires in the
subsystem. It is equivalent to an integer with value -1."""


class Operator(tq.QuantumModule):
    fixed_ops = [
        'Hadamard',
        'SHadamard',
        'PauliX',
        'PauliY',
        'PauliZ',
        'I',
        'S',
        'T',
        'SX',
        'CNOT',
        'CZ',
        'CY',
        'SWAP',
        'SSWAP',
        'CSWAP',
        'Toffoli',
        'MultiCNOT',
        'MultiXCNOT',
        'Reset',
    ]

    parameterized_ops = [
        'RX',
        'RY',
        'RZ',
        'RXX',
        'RYY',
        'RZZ',
        'RZX',
        'PhaseShift',
        'Rot',
        'MultiRZ',
        'CRX',
        'CRY',
        'CRZ',
        'CRot',
        'U1',
        'U2',
        'U3',
        'CU1',
        'CU2',
        'CU3',
        'QubitUnitary',
        'QubitUnitaryFast',
        'TrainableUnitary',
        'TrainableUnitaryStrict',
    ]

    @property
    def name(self):
        """String for the name of the operator."""
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    def __init__(self,
                 has_params: bool = False,
                 trainable: bool = False,
                 init_params=None,
                 n_wires=None,
                 wires=None):
        super().__init__()
        self.params = None
        # number of wires of the operator
        # n_wires is used in gates that can be applied to arbitrary number
        # of qubits such as MultiRZ
        self.n_wires = n_wires
        # wires that the operator applies to
        self.wires = wires
        self._name = self.__class__.__name__
        # for static mode
        self.static_matrix = None
        self.inverse = False

        try:
            assert not (trainable and not has_params)
        except AssertionError:
            has_params = True
            logger.warning(f"Module must have parameters to be trainable; "
                           f"Switched 'has_params' to True.")

        self.has_params = has_params
        self.trainable = trainable
        if self.has_params:
            self.params = self.build_params(trainable=self.trainable)
            self.reset_params(init_params)

    @classmethod
    def _matrix(cls, params):
        raise NotImplementedError

    @property
    def matrix(self):
        return self._matrix(self.params)

    @classmethod
    def _eigvals(cls, params):
        raise NotImplementedError

    @property
    def eigvals(self):
        return self._eigvals(self.params)

    def _get_unitary_matrix(self):
        return self.matrix

    def set_wires(self, wires):
        self.wires = [wires] if isinstance(wires, int) else wires

    def forward(self, q_device: tq.QuantumDevice, wires=None, params=None,
                inverse=False):
        # try:
        #     assert self.name in self.fixed_ops or \
        #            self.has_params ^ (params is not None)
        # except AssertionError as err:
        #     logger.exception(f"Parameterized gate either has its "
        #                      f"own parameters or has input as parameters")
        #     raise err

        # try:
        #     assert not (self.wires is None and wires is None)
        # except AssertionError as err:
        #     logger.exception(f"Need to specify the wires either when "
        #                      f"initialize or when forward")
        #     raise err

        if params is not None:
            self.params = params

        if self.params is not None:
            self.params = self.params.unsqueeze(-1) if self.params.dim() == 1 \
                else self.params

        if wires is not None:
            # update the wires
            wires = [wires] if isinstance(wires, int) else wires
            self.wires = wires

        self.inverse = inverse

        if self.static_mode:
            self.parent_graph.add_op(self)
            return

        # non-parameterized gate
        if self.params is None:
            if self.n_wires is None:
                self.func(q_device, self.wires, inverse=inverse)
            else:
                self.func(q_device, self.wires, n_wires=self.n_wires,
                          inverse=inverse)
        else:
            if isinstance(self.noise_model_tq, tq.NoiseModelTQPhase):
                params = self.noise_model_tq.add_noise(self.params)
            else:
                params = self.params

            if self.n_wires is None:
                self.func(q_device, self.wires, params=params,
                          inverse=inverse)
            else:
                self.func(q_device, self.wires, params=params,
                          n_wires=self.n_wires, inverse=inverse)

        if self.noise_model_tq is not None and \
                self.noise_model_tq.is_add_noise:
            noise_ops = self.noise_model_tq.sample_noise_op(self)
            if len(noise_ops):
                for noise_op in noise_ops:
                    noise_op(q_device)


class Observable(Operator, metaclass=ABCMeta):
    def __init__(self,
                 has_params: bool = False,
                 trainable: bool = False,
                 init_params=None,
                 n_wires=None,
                 wires=None):
        super().__init__(
            has_params=has_params,
            trainable=trainable,
            init_params=init_params,
            n_wires=n_wires,
            wires=wires
        )
        self.return_type = None

    def diagonalizing_gates(self):
        raise NotImplementedError


class Operation(Operator, metaclass=ABCMeta):
    def __init__(self,
                 has_params: bool = False,
                 trainable: bool = False,
                 init_params=None,
                 n_wires=None,
                 wires=None):
        super().__init__(
            has_params=has_params,
            trainable=trainable,
            init_params=init_params,
            n_wires=n_wires,
            wires=wires
        )

    @property
    def matrix(self):
        op_matrix = self._matrix(self.params)

        return op_matrix

    @property
    def eigvals(self):
        op_eigvals = self._eigvals(self.params)

        return op_eigvals

    def init_params(self):
        raise NotImplementedError

    def build_params(self, trainable):
        parameters = nn.Parameter(torch.empty([1, self.num_params],
                                              dtype=F_DTYPE))
        parameters.requires_grad = True if trainable else False
        self.register_parameter(f"{self.name}_params", parameters)
        return parameters

    def reset_params(self, init_params=None):
        if init_params is not None:
            if isinstance(init_params, Iterable):
                for k, init_param in enumerate(init_params):
                    torch.nn.init.constant_(self.params[:, k], init_param)
            else:
                torch.nn.init.constant_(self.params, init_params)
        else:
            torch.nn.init.uniform_(self.params, -np.pi, np.pi)


class DiagonalOperation(Operation, metaclass=ABCMeta):
    @classmethod
    def _eigvals(cls, params):
        raise NotImplementedError

    @property
    def eigvals(self):
        return super().eigvals

    @classmethod
    def _matrix(cls, params):
        return torch.diag(cls._eigvals(params))


class Hadamard(Observable, metaclass=ABCMeta):
    num_params = 0
    num_wires = 1
    eigvals = torch.tensor([1, -1], dtype=C_DTYPE)
    matrix = mat_dict['hadamard']
    func = staticmethod(tqf.hadamard)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix

    @classmethod
    def _eigvals(cls, params):
        return cls.eigvals

    def diagonalizing_gates(self):
        return [tq.RY(has_params=True,
                      trainable=False,
                      init_params=-np.pi / 4)]


class SHadamard(Operation, metaclass=ABCMeta):
    num_params = 0
    num_wires = 1
    matrix = mat_dict['shadamard']
    func = staticmethod(tqf.shadamard)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix


class PauliX(Observable, metaclass=ABCMeta):
    num_params = 0
    num_wires = 1
    eigvals = torch.tensor([1, -1], dtype=C_DTYPE)
    matrix = mat_dict['paulix']
    func = staticmethod(tqf.paulix)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix

    @classmethod
    def _eigvals(cls, params):
        return cls.eigvals

    def diagonalizing_gates(self):
        return [tq.Hadamard()]


class PauliY(Observable, metaclass=ABCMeta):
    num_params = 0
    num_wires = 1
    eigvals = torch.tensor([1, -1], dtype=C_DTYPE)
    matrix = mat_dict['pauliy']
    func = staticmethod(tqf.pauliy)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix

    @classmethod
    def _eigvals(cls, params):
        return cls.eigvals

    def diagonalizing_gates(self):
        return [tq.PauliZ(), tq.S(), tq.Hadamard()]


class PauliZ(Observable, metaclass=ABCMeta):
    num_params = 0
    num_wires = 1
    eigvals = torch.tensor([1, -1], dtype=C_DTYPE)
    matrix = mat_dict['pauliz']
    func = staticmethod(tqf.pauliz)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix

    @classmethod
    def _eigvals(cls, params):
        return cls.eigvals

    def diagonalizing_gates(self):
        return []


class I(Observable, metaclass=ABCMeta):
    num_params = 0
    num_wires = 1
    eigvals = torch.tensor([1, 1], dtype=C_DTYPE)
    matrix = mat_dict['i']
    func = staticmethod(tqf.i)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix

    @classmethod
    def _eigvals(cls, params):
        return cls.eigvals

    def diagonalizing_gates(self):
        return []


class S(DiagonalOperation, metaclass=ABCMeta):
    num_params = 0
    num_wires = 1
    eigvals = torch.tensor([1, 1j], dtype=C_DTYPE)
    matrix = mat_dict['s']
    func = staticmethod(tqf.s)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix

    @classmethod
    def _eigvals(cls, params):
        return cls.eigvals


class T(DiagonalOperation, metaclass=ABCMeta):
    num_params = 0
    num_wires = 1
    eigvals = torch.tensor([1, 1j], dtype=C_DTYPE)
    matrix = mat_dict['t']
    func = staticmethod(tqf.t)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix

    @classmethod
    def _eigvals(cls, params):
        return cls.eigvals


class SX(Operation, metaclass=ABCMeta):
    num_params = 0
    num_wires = 1
    eigvals = torch.tensor([1, 1j], dtype=C_DTYPE)
    matrix = mat_dict['sx']
    func = staticmethod(tqf.sx)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix

    @classmethod
    def _eigvals(cls, params):
        return cls.eigvals


class CNOT(Operation, metaclass=ABCMeta):
    num_params = 0
    num_wires = 2
    matrix = mat_dict['cnot']
    func = staticmethod(tqf.cnot)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix


class CZ(DiagonalOperation, metaclass=ABCMeta):
    num_params = 0
    num_wires = 2
    eigvals = np.array([1, 1, 1, -1])
    matrix = mat_dict['cz']
    func = staticmethod(tqf.cz)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix

    @classmethod
    def _eigvals(cls, params):
        return cls.eigvals


class CY(Operation, metaclass=ABCMeta):
    num_params = 0
    num_wires = 2
    matrix = mat_dict['cy']
    func = staticmethod(tqf.cy)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix


class SWAP(Operation, metaclass=ABCMeta):
    num_params = 0
    num_wires = 2
    matrix = mat_dict['swap']
    func = staticmethod(tqf.swap)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix


class SSWAP(Operation, metaclass=ABCMeta):
    num_params = 0
    num_wires = 2
    matrix = mat_dict['sswap']
    func = staticmethod(tqf.sswap)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix


class CSWAP(Operation, metaclass=ABCMeta):
    num_params = 0
    num_wires = 3
    matrix = mat_dict['cswap']
    func = staticmethod(tqf.cswap)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix


class Toffoli(Operation, metaclass=ABCMeta):
    num_params = 0
    num_wires = 3
    matrix = mat_dict['toffoli']
    func = staticmethod(tqf.toffoli)

    @classmethod
    def _matrix(cls, params):
        return cls.matrix


class RX(Operation, metaclass=ABCMeta):
    num_params = 1
    num_wires = 1
    func = staticmethod(tqf.rx)

    @classmethod
    def _matrix(cls, params):
        return tqf.rx_matrix(params)


class RY(Operation, metaclass=ABCMeta):
    num_params = 1
    num_wires = 1
    func = staticmethod(tqf.ry)

    @classmethod
    def _matrix(cls, params):
        return tqf.ry_matrix(params)


class RZ(DiagonalOperation, metaclass=ABCMeta):
    num_params = 1
    num_wires = 1
    func = staticmethod(tqf.rz)

    @classmethod
    def _matrix(cls, params):
        return tqf.rz_matrix(params)


class PhaseShift(DiagonalOperation, metaclass=ABCMeta):
    num_params = 1
    num_wires = 1
    func = staticmethod(tqf.phaseshift)

    @classmethod
    def _matrix(cls, params):
        return tqf.phaseshift_matrix(params)


class Rot(Operation, metaclass=ABCMeta):
    num_params = 3
    num_wires = 1
    func = staticmethod(tqf.rot)

    @classmethod
    def _matrix(cls, params):
        return tqf.rot_matrix(params)


class MultiRZ(DiagonalOperation, metaclass=ABCMeta):
    num_params = 1
    num_wires = AnyWires
    func = staticmethod(tqf.multirz)

    @classmethod
    def _matrix(cls, params, n_wires):
        return tqf.multirz_matrix(params, n_wires)


class RXX(Operation, metaclass=ABCMeta):
    num_params = 1
    num_wires = 2
    func = staticmethod(tqf.rxx)

    @classmethod
    def _matrix(cls, params):
        return tqf.rxx_matrix(params)


class RYY(Operation, metaclass=ABCMeta):
    num_params = 1
    num_wires = 2
    func = staticmethod(tqf.ryy)

    @classmethod
    def _matrix(cls, params):
        return tqf.ryy_matrix(params)


class RZZ(DiagonalOperation, metaclass=ABCMeta):
    num_params = 1
    num_wires = 2
    func = staticmethod(tqf.rzz)

    @classmethod
    def _matrix(cls, params):
        return tqf.rzz_matrix(params)


class RZX(Operation, metaclass=ABCMeta):
    num_params = 1
    num_wires = 2
    func = staticmethod(tqf.rzx)

    @classmethod
    def _matrix(cls, params):
        return tqf.rzx_matrix(params)


class TrainableUnitary(Operation, metaclass=ABCMeta):
    num_params = AnyNParams
    num_wires = AnyWires
    func = staticmethod(tqf.qubitunitaryfast)

    def build_params(self, trainable):
        parameters = nn.Parameter(torch.empty(
            1, 2 ** self.n_wires, 2 ** self.n_wires, dtype=C_DTYPE))
        parameters.requires_grad = True if trainable else False
        self.register_parameter(f"{self.name}_params", parameters)
        return parameters

    def reset_params(self, init_params=None):
        mat = torch.randn((1, 2 ** self.n_wires, 2 ** self.n_wires),
                          dtype=C_DTYPE)
        U, Sigma, V = torch.svd(mat)
        self.params.data.copy_(U.matmul(V.permute(0, 2, 1)))

    @staticmethod
    def _matrix(self, params):
        return tqf.qubitunitaryfast(params)


class TrainableUnitaryStrict(TrainableUnitary, metaclass=ABCMeta):
    num_params = AnyNParams
    num_wires = AnyWires
    func = staticmethod(tqf.qubitunitarystrict)


class CRX(Operation, metaclass=ABCMeta):
    num_params = 1
    num_wires = 2
    func = staticmethod(tqf.crx)

    @classmethod
    def _matrix(cls, params):
        return tqf.crx_matrix(params)


class CRY(Operation, metaclass=ABCMeta):
    num_params = 1
    num_wires = 2
    func = staticmethod(tqf.cry)

    @classmethod
    def _matrix(cls, params):
        return tqf.cry_matrix(params)


class CRZ(Operation, metaclass=ABCMeta):
    num_params = 1
    num_wires = 2
    func = staticmethod(tqf.crz)

    @classmethod
    def _matrix(cls, params):
        return tqf.crz_matrix(params)


class CRot(Operation, metaclass=ABCMeta):
    num_params = 3
    num_wires = 2
    func = staticmethod(tqf.crot)

    @classmethod
    def _matrix(cls, params):
        return tqf.crot_matrix(params)


class U1(DiagonalOperation, metaclass=ABCMeta):
    # U1 is the same as phaseshift
    num_params = 1
    num_wires = 1
    func = staticmethod(tqf.u1)

    @classmethod
    def _matrix(cls, params):
        return tqf.u1_matrix(params)


class CU1(DiagonalOperation, metaclass=ABCMeta):
    num_params = 1
    num_wires = 2
    func = staticmethod(tqf.cu1)

    @classmethod
    def _matrix(cls, params):
        return tqf.cu1_matrix(params)


class U2(Operation, metaclass=ABCMeta):
    num_params = 2
    num_wires = 1
    func = staticmethod(tqf.u2)

    @classmethod
    def _matrix(cls, params):
        return tqf.u2_matrix(params)


class CU2(Operation, metaclass=ABCMeta):
    num_params = 2
    num_wires = 2
    func = staticmethod(tqf.cu2)

    @classmethod
    def _matrix(cls, params):
        return tqf.cu2_matrix(params)


class U3(Operation, metaclass=ABCMeta):
    num_params = 3
    num_wires = 1
    func = staticmethod(tqf.u3)

    @classmethod
    def _matrix(cls, params):
        return tqf.u3_matrix(params)


class CU3(Operation, metaclass=ABCMeta):
    num_params = 3
    num_wires = 2
    func = staticmethod(tqf.cu3)

    @classmethod
    def _matrix(cls, params):
        return tqf.cu3_matrix(params)


class QubitUnitary(Operation, metaclass=ABCMeta):
    num_params = AnyNParams
    num_wires = AnyWires
    func = staticmethod(tqf.qubitunitary)

    @classmethod
    def _matrix(cls, params):
        return tqf.qubitunitary(params)

    def build_params(self, trainable):
        return None

    def reset_params(self, init_params=None):
        self.params = torch.tensor(init_params, dtype=C_DTYPE)
        self.register_buffer(f"{self.name}_unitary", self.params)


class QubitUnitaryFast(Operation, metaclass=ABCMeta):
    num_params = AnyNParams
    num_wires = AnyWires
    func = staticmethod(tqf.qubitunitaryfast)

    @classmethod
    def _matrix(cls, params):
        return tqf.qubitunitaryfast(params)

    def build_params(self, trainable):
        return None

    def reset_params(self, init_params=None):
        self.params = torch.tensor(init_params, dtype=C_DTYPE)
        self.register_buffer(f"{self.name}_unitary", self.params)


class MultiCNOT(Operation, metaclass=ABCMeta):
    num_params = 0
    num_wires = AnyWires
    func = staticmethod(tqf.multicnot)

    @classmethod
    def _matrix(cls, params, n_wires):
        return tqf.multicnot_matrix(n_wires)

    @property
    def matrix(self):
        op_matrix = self._matrix(self.params, self.n_wires)
        return op_matrix


class MultiXCNOT(Operation, metaclass=ABCMeta):
    num_params = 0
    num_wires = AnyWires
    func = staticmethod(tqf.multixcnot)

    @classmethod
    def _matrix(cls, params, n_wires):
        return tqf.multixcnot_matrix(n_wires)

    @property
    def matrix(self):
        op_matrix = self._matrix(self.params, self.n_wires)
        return op_matrix


class Reset(Operator, metaclass=ABCMeta):
    num_params = 0
    num_wires = AnyWires
    func = staticmethod(tqf.reset)

    @classmethod
    def _matrix(cls, params):
        return None


op_name_dict = {
    'hadamard': Hadamard,
    'h': Hadamard,
    'shadamard': SHadamard,
    'sh': SHadamard,
    'paulix': PauliX,
    'x': PauliX,
    'pauliy': PauliY,
    'y': PauliY,
    'pauliz': PauliZ,
    'z': PauliZ,
    'i': I,
    's': S,
    't': T,
    'sx': SX,
    'cx': CNOT,
    'cnot': CNOT,
    'cz': CZ,
    'cy': CY,
    'rx': RX,
    'ry': RY,
    'rz': RZ,
    'rxx': RXX,
    'xx': RXX,
    'ryy': RYY,
    'yy': RYY,
    'rzz': RZZ,
    'zz': RZZ,
    'rzx': RZX,
    'zx': RZX,
    'swap': SWAP,
    'sswap': SSWAP,
    'cswap': CSWAP,
    'toffoli': Toffoli,
    'ccx': Toffoli,
    'phaseshift': PhaseShift,
    'rot': Rot,
    'multirz': MultiRZ,
    'crx': CRX,
    'cry': CRY,
    'crz': CRZ,
    'crot': CRot,
    'u1': U1,
    'p': U1,
    'u2': U2,
    'u3': U3,
    'u': U3,
    'cu1': CU1,
    'cp': CU1,
    'cu2': CU2,
    'cu3': CU3,
    'cu': CU3,
    'qubitunitary': QubitUnitary,
    'qubitunitarystrict': QubitUnitaryFast,
    'qubitunitaryfast': QubitUnitaryFast,
    'trainableunitary': TrainableUnitary,
    'trainableunitarystrict': TrainableUnitaryStrict,
    'multicnot': MultiCNOT,
    'multixcnot': MultiXCNOT,
    'reset': Reset,
}
