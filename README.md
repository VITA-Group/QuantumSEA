# QuantumSEA: In-Time Sparse Exploration for Noise Adaptive Quantum Circuits

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Codes for this paper [QuantumSEA: In-Time Sparse Exploration for Noise Adaptive Quantum Circuits](https://www.jqgu.net/publications/papers/Quant_QCE2023_Gu.pdf) [QCE 2023]

## Overview

Parameterized Quantum Circuits (PQC) have obtained increasing popularity thanks to their great potential for near-term Noisy Intermediate-Scale Quantum (NISQ) computers. Achieving quantum advantages usually requires a large number of qubits and quantum circuits with enough capacity. However, limited coherence time and massive quantum noises severely constrain the size of quantum circuits that can be executed reliably on real machines. To address these two pain points, we propose QuantumSEA, an in-time sparse exploration for noise-adaptive quantum circuits, aiming to achieve two key objectives: (1) implicit circuits capacity during training - by dynamically exploring the circuit’s sparse connectivity and sticking a fixed small number of quantum gates throughout the training which satisfies the coherence time and enjoy light noises, enabling feasible executions on real quantum devices; (2) noise robustness - by jointly optimizing the topology and parameters of quantum circuits under real device noise models. In each update step of sparsity, we leverage the moving average of historical gradients to grow necessary gates and utilize salience-based pruning to eliminate insignificant gates. Extensive experiments are conducted with 7 Quantum Machine Learning (QML) and Variational Quantum Eigensolver (VQE) benchmarks on 6 simulated or real quantum computers, where QuantumSEA consistently surpasses noise-aware search, humandesigned, and randomly generated quantum circuit baselines by a clear performance margin. For example, even in the most challenging on-chip training regime, our method establishes stateof-the-art results with only half the number of quantum gates and ∼ 2× time saving of circuit executions.

## Prerequisites

```
git clone https://github.com/mit-han-lab/torchquantum.git
cd torchquantum
pip install --editable .
python fix_qiskit_parameterization.py
```

## Usage

```
# F-MNIST2, ibmq_quito, RXYZ circuit
bash script/fmnist2/Dense.sh # Dense
bash script/fmnist2/Human.sh # Human Design 50% gates
bash script/fmnist2/Random.sh # Random Design 50% gates
bash script/fmnist2/SEA.sh # Our methods 50% gates

# MNIST2, ibmq_quito, RXYZ circuit
bash script/mnist2/Dense.sh # Dense
bash script/mnist2/Human.sh # Human Design 50% gates
bash script/mnist2/Random.sh # Random Design 50% gates
bash script/mnist2/SEA.sh # Our methods 50% gates

```

## Acknowledgement

This repo is based on https://github.com/mit-han-lab/torchquantum

## Citation

```
@article{chenquantumsea,
  title={QuantumSEA: In-Time Sparse Exploration for Noise Adaptive Quantum Circuits},
  author={Chen, Tianlong and Zhang, Zhenyu and Wang, Hanrui and Gu, Jiaqi and Li, Zirui and Pan, David Z and Chong, Frederic T and Han, Song and Wang, Zhangyang}
}
```

