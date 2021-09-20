# PyTorch implementation of NAQS and RNN wave functions

This repo contains my simple PyTorch implementation of Neural Autoregressive Quantum States (NAQS) in [Phys. Rev. Lett. 124, 020503](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.124.020503) and Recurrent Neural Network wave functions (RNN wave functions) in [Phys. Rev. Research 2, 023358](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.023358).

## Dependencies

The code is tested on Tesla V100 with `Python 3.6.13`, `PyTorch 1.9.0` and `CUDA 10.2`.

## TODO

1. Multi-layer and 2D RNN, PixelCNN-like architecture described in original papers.

2. Implementing symmetries.

## Reference:

1. [netket/netket](https://github.com/netket/netket)

2. [mhibatallah/RNNWavefunctions](https://github.com/mhibatallah/RNNWavefunctions)

3. [kafischer/neural-quantum-states](https://github.com/kafischer/neural-quantum-states)