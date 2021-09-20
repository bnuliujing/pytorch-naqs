import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

from ham import Ising1D, Ising2D, J1J2Chain
from model.rnn import PositiveRNN, ComplexRNN
from model.naqs import PositiveNAQS, ComplexNAQS


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group('physical params')
    group.add_argument('--ham', type=str, choices=['ising', 'j1j2'], help='hamiltonian')
    group.add_argument('--dim', type=int, choices=[1, 2], help='dimension')
    group.add_argument('--pbc', action='store_true', help='use PBC')
    group.add_argument('--L', type=int, default=10, help='length, default: 10')
    group.add_argument('--h', type=float, default=1.0, help='external field h for TFIM, default: 1.0')
    group.add_argument('--J2', type=float, default=0.2, help='J2 for J1-J2 model, default: 0.2')

    group = parser.add_argument_group('training params')
    group.add_argument('--batch-size', type=int, default=10000, help='batch size, default: 10000')
    group.add_argument('--epochs', type=int, default=10000, help='number of epochs to train, default: 10000')
    group.add_argument('--lr', type=float, default=1e-3, help='leanring rate, default: 1e-3')
    group.add_argument('--seed', type=float, default=2050, help='random seed, default: 2050')
    group.add_argument('--gpu', type=str, default='0', help='default gpu id, default: 0')
    group.add_argument('--no-cuda', action='store_true', default=False, help='disable cuda')

    group = parser.add_argument_group('neural-network model params')
    group.add_argument('--model', type=str, choices=['rnn', 'conv'], help='choose rnn or conv model')
    group.add_argument('--net-type', type=str, choices=['real', 'complex'], help='use real or complex parameterization')
    group.add_argument('--net-depth', type=int, default=3, help='network depth in Conv, default: 3')
    group.add_argument('--hidden-size',
                       type=int,
                       default=10,
                       help='hidden size in GRU or channels in Conv, default: 10')
    group.add_argument('--kernel-size', type=int, default=11, help='kernel size, default: 11')
    group.add_argument('--padding', type=int, default=5, help='padding, default: 5')

    group = parser.add_argument_group('other params')
    group.add_argument('--save-model', action='store_true', default=False, help='save model')
    group.add_argument('--output-dir', default='./saved/tmp', help='output folder')

    args = parser.parse_args()
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device('cuda:' + str(args.gpu) if not args.no_cuda else 'cpu')

    torch.manual_seed(args.seed)

    # initialize neural-network model
    if args.model == 'rnn':
        if args.net_type == 'real':
            model = PositiveRNN(**vars(args)).to(args.device)
        else:
            model = ComplexRNN(**vars(args)).to(args.device)
    elif args.model == 'conv':
        if args.net_type == 'real':
            model = PositiveNAQS(**vars(args)).to(args.device)
        else:
            model = ComplexNAQS(**vars(args)).to(args.device)

    print(model)
    params = sum([np.prod(p.size()) for p in model.parameters()])
    print('\nTotal number of parameters: %i' % params)
    time.sleep(2)

    # hamiltonian
    if args.dim == 1 and args.ham == 'ising':
        ham = Ising1D(**vars(args))
    elif args.dim == 2 and args.ham == 'ising':
        ham = Ising2D(**vars(args))
    elif args.dim == 1 and args.ham == 'j1j2':
        ham = J1J2Chain(**vars(args))
    else:
        raise ValueError('Hamiltonian not implemented.')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # training
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        with torch.no_grad():
            samples = model.sample()
            E_loc = model.local_energy(samples, ham)
        log_psi = model.log_psi(samples)
        if args.net_type == 'real':
            loss_reinforce = torch.mean(log_psi * (E_loc - E_loc.mean()))
        else:
            loss_reinforce = torch.real(torch.mean(log_psi.conj() * (E_loc - E_loc.mean())))
        loss_reinforce.backward()
        optimizer.step()
        with torch.no_grad():
            energy_mean = E_loc.mean().item() if args.net_type == 'real' else E_loc.real.mean().item()
            energy_var = E_loc.var().item() if args.net_type == 'real' else E_loc.real.var().item()
            print('Epoch: {:>5d}\tmean(E): {:+.6f}\tvar(E): {:.4f}'.format(epoch, energy_mean, energy_var))
            with open(os.path.join(args.output_dir, 'log.txt'), 'a', newline='\n') as f:
                f.write('{:d} {:.10f} {:.10f}\n'.format(epoch, energy_mean, energy_var))

    if args.save_model:
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'model.pt'))
    print('\nEstimating final variational energy...')
    E_est = []
    with torch.no_grad():
        for _ in range(100):  # use total 100 * batch_size samples to calculate the variational energy
            samples = model.sample()
            E_loc = model.local_energy(samples, ham)
            energy_mean = E_loc.mean().item() if args.net_type == 'real' else E_loc.real.mean().item()
            energy_var = E_loc.var().item() if args.net_type == 'real' else E_loc.real.var().item()
            with open(os.path.join(args.output_dir, 'estimate.txt'), 'a', newline='\n') as f:
                f.write('{:.10f} {:.10f}\n'.format(energy_mean, energy_var))
            E_est.append(energy_mean)
    print('E = {:.10f} \u00B1 {:.10f}'.format(np.mean(E_est), np.std(E_est) / math.sqrt(100)))


if __name__ == '__main__':
    main()
