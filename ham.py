import math
from abc import ABC, abstractmethod

import numpy as np
import torch


class Hamiltonian(ABC):
    @abstractmethod
    def find_matrix_elements(self, state):
        pass


class Ising1D(Hamiltonian):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.L = kwargs['L']
        self.h = kwargs['h']
        self.pbc = kwargs['pbc']
        self.device = kwargs['device']

    def find_matrix_elements(self, state):
        """find non-zero matrix elements of <x|H|x`>

        Args:
            state (Tensor): quantum state, (batch_size, L), values in {-1, +1}

        Returns:
            Tensor: non-zero matrix element, (batch_size, L+1) for 1D TFI model
        """
        batch_size = state.shape[0]
        matrix_elements = -self.h * torch.ones((batch_size, self.L + 1), device=self.device)

        matrix_elements[:, 0] = -1 * torch.sum(state[:, :self.L - 1] * state[:, 1:], dim=1)
        if self.pbc:
            matrix_elements[:, 0] -= state[:, -1] * state[:, 0]

        return matrix_elements

    def exact(self):
        """Exact solution of ground state energy
        """
        print('\nExact Diagonalization...')
        n_total = int(math.pow(2, self.L))
        H = np.zeros((n_total, n_total))
        for idx in range(n_total):
            s = np.binary_repr(idx, width=self.L)

            # digonal element is -\sum s_i s_j
            state = np.array(list(s)).astype(np.float64) * 2 - 1

            H_ss = 0.
            for i in range(self.L - 1):
                H_ss -= state[i] * state[i + 1]
            if self.pbc:
                H_ss -= state[0] * state[-1]
            H[idx, idx] = H_ss

            # find non-zero index with value -h
            for i in range(self.L):
                bin_state = np.array(list(s)).astype(np.int)
                # flip spin
                bin_state[i] = 1 - bin_state[i]
                j = bin_state.dot(1 << np.arange(bin_state.size)[::-1])
                H[idx, j] = -self.h
                # flip back
                bin_state[i] = 1 - bin_state[i]

        print('Enumeration done! Calculating the eigen values...')
        ground_state_energy = np.min(np.linalg.eigvals(H))
        print('Ground state energy: %.10f' % ground_state_energy.real)


class Ising2D(Hamiltonian):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.L = kwargs['L']
        self.h = kwargs['h']
        self.pbc = kwargs['pbc']
        self.device = kwargs['device']

    def find_matrix_elements(self, state):
        """find non-zero matrix elements of <x|H|x`>

        Args:
            state (Tensor): quantum state, (batch_size, L, L), values in {-1, +1}

        Returns:
            Tensor: non-zero matrix element, (batch_size, L^2+1) for 1D TFI model
        """
        batch_size = state.shape[0]
        matrix_elements = -self.h * torch.ones((batch_size, self.L**2 + 1), device=self.device)

        matrix_elements[:, 0] = -1 * torch.sum(state[:, :, :self.L - 1] * state[:, :, 1:], dim=(1, 2))
        matrix_elements[:, 0] -= torch.sum(state[:, :self.L - 1, :] * state[:, 1:, :], dim=(1, 2))
        if self.pbc:
            matrix_elements[:, 0] -= state[:, :, -1] * state[:, :, 0]
            matrix_elements[:, 0] -= state[:, -1, :] * state[:, 0, :]

        return matrix_elements


class J1J2Chain(Hamiltonian):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.L = kwargs['L']
        self.J2 = kwargs['J2']
        self.pbc = kwargs['pbc']
        self.device = kwargs['device']

    def find_matrix_elements(self, state):
        """find non-zero matrix elements of <x|H|x`>
        In TFIM, the number of non-zero elements is fixed (L+1 or L^2+1)
        This is not the case in J1-J2 model. But we can take the maximum length (2*L+1)
        Order: [diagonal term, ***J1 term from L to R***, ***J2 term from L to R***]

        Args:
            state (Tensor): quantum state, (batch_size, L), values in {-1, +1}

        Returns:
            Tensor: non-zero matrix element, (batch_size, L*2+1)
        """
        batch_size = state.shape[0]
        matrix_elements = torch.zeros((batch_size, self.L * 2 + 1), device=self.device)
        # J1 part
        spin_mul = state[:, :self.L - 1] * state[:, 1:]
        matrix_elements[:, 0] = 0.25 * torch.sum(spin_mul, dim=1)  # diagonal part
        spin_mul[spin_mul == -1.] = 0.5
        spin_mul[spin_mul == 1.] = 0.
        matrix_elements[:, 1:self.L] = spin_mul  # non-diagonal part
        if self.pbc:
            spin_mul = state[:, -1] * state[:, 0]
            matrix_elements[:, 0] += 0.25 * spin_mul  # diagonal part
            spin_mul[spin_mul == -1.] = 0.5
            spin_mul[spin_mul == 1.] = 0.
            matrix_elements[:, self.L] = spin_mul  # non-diagonal part
        # J2 part
        spin_mul = state[:, :self.L - 2] * state[:, 2:]
        matrix_elements[:, 0] += 0.25 * self.J2 * torch.sum(spin_mul, dim=1)  # diagonal part
        spin_mul[spin_mul == -1] = 0.5 * self.J2
        spin_mul[spin_mul == 1] = 0.
        matrix_elements[:, self.L + 1:self.L * 2 - 1] = spin_mul  # non-diagonal part
        if self.pbc:
            spin_mul = torch.stack([state[:, -2] * state[:, 0], state[:, -1] * state[:, 1]], dim=1)
            matrix_elements[:, 0] += 0.25 * self.J2 * torch.sum(spin_mul, dim=1)  # diagonal part
            spin_mul[spin_mul == -1.] = 0.5 * self.J2
            spin_mul[spin_mul == 1.] = 0.
            matrix_elements[:, -2:] = spin_mul  # non-diagonal part

        return matrix_elements


if __name__ == '__main__':
    # # test
    # arg_dict = {'L': 10, 'h': 2, 'pbc': False, 'device': 'cpu'}
    # H = Ising1D(**arg_dict)
    # H.exact()

    # test
    arg_dict = {'L': 10, 'J2': 0.2, 'pbc': True, 'device': 'cpu'}
    H = J1J2Chain(**arg_dict)
    state = torch.randint(0, 2, size=(8, arg_dict['L']), dtype=torch.float) * 2 - 1
    print(state)
    matrix_elements = H.find_matrix_elements(state)
    print(matrix_elements)
