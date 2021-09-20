import math

import torch
import torch.nn as nn


class RealBaseModel(nn.Module):
    """
    one has to implement forward(), log_prob() and sample() method
    """
    def forward(self):
        pass

    def log_prob(self):
        pass

    def sample(self):
        pass

    def log_psi(self, x):
        """log Psi(s) = 1/2 log P(s)

        Args:
            x (Tensor): quantum state, (batch_size, L), values in {-1, +1}

        Returns:
            Tensor: log amplitude, (batch_size, )
        """
        return self.log_prob(x) / 2

    def local_energy(self, x, ham):
        """for a given state x, find its non-zero matrix elements <x|H|x`> 
        and the coresponding amplitude ratio Psi(x)/Psi(x`) and then calculate the local energy E_loc

        Args:
            x (Tensor): quantum state, (batch_size, L) or (batch_size, L, L), values in {-1, +1}
            ham (Hamiltonian): specific hamiltonian

        Returns:
            Tensor: local energy E_loc, (batch_size, )
        """
        matrix_elements = ham.find_matrix_elements(x)
        amplitude_ratio = self.amplitude_ratio(x)
        E_loc = torch.sum(matrix_elements * amplitude_ratio, dim=1)

        return E_loc

    def log_amplitude_ratio(self, x):
        """for a given state x, calculate its log amplitude ratio log psi(x`) - log psi(x)

        Args:
            x (Tensor): quantum state, (batch_size, L) or (batch_size, L, L), values in {-1, +1}

        Returns:
            Tensor: log amplitude ratio, (batch_size, *)
        """
        if self.ham == 'ising':
            if self.dim == 1:
                log_amplitude_ratio = torch.zeros((self.batch_size, self.L + 1), dtype=torch.float, device=self.device)
                log_psi = self.log_psi(x)
                for i in range(self.L):
                    # flip spin
                    x[:, i] *= -1
                    log_psi_prime = self.log_psi(x)
                    log_amplitude_ratio[:, i + 1] = log_psi_prime - log_psi
                    # flip back
                    x[:, i] *= -1
            elif self.dim == 2:
                log_amplitude_ratio = torch.zeros((self.batch_size, self.L**2 + 1),
                                                  dtype=torch.float,
                                                  device=self.device)
                log_psi = self.log_psi(x)
                for i in range(self.L):
                    for j in range(self.L):
                        # flip spin
                        x[:, i, j] *= -1
                        log_psi_prime = self.log_psi(x)
                        log_amplitude_ratio[:, i * self.L + j + 1] = log_psi_prime - log_psi
                        # flip back
                        x[:, i, j] *= -1
        elif self.ham == 'j1j2':
            log_amplitude_ratio = -torch.zeros((self.batch_size, self.L * 2 + 1), dtype=torch.float, device=self.device)
            log_psi = self.log_psi(x)
            # J1 part
            for i in range(self.L - 1):
                # flip two spins (nearest)
                x[:, i] *= -1
                x[:, i + 1] *= -1
                log_psi_prime = self.log_psi(x)
                log_amplitude_ratio[:, i + 1] = log_psi_prime - log_psi
                # flip back
                x[:, i] *= -1
                x[:, i + 1] *= -1
            if self.pbc:
                # flip two spins (nearest)
                x[:, -1] *= -1
                x[:, 0] *= -1
                log_psi_prime = self.log_psi(x)
                log_amplitude_ratio[:, self.L] = log_psi_prime - log_psi
                # flip back
                x[:, -1] *= -1
                x[:, 0] *= -1
            # J2 part
            for i in range(self.L - 2):
                # flip two spins (next nearest)
                x[:, i] *= -1
                x[:, i + 2] *= -1
                log_psi_prime = self.log_psi(x)
                log_amplitude_ratio[:, i + self.L + 1] = log_psi_prime - log_psi
                # flip back
                x[:, i] *= -1
                x[:, i + 2] *= -1
            if self.pbc:
                # flip two spins (next nearest)
                x[:, -2] *= -1
                x[:, 0] *= -1
                log_psi_prime = self.log_psi(x)
                log_amplitude_ratio[:, -2] = log_psi_prime - log_psi
                # flip back
                x[:, -2] *= -1
                x[:, 0] *= -1
                # flip two spins (next nearest)
                x[:, -1] *= -1
                x[:, 1] *= -1
                log_psi_prime = self.log_psi(x)
                log_amplitude_ratio[:, -1] = log_psi_prime - log_psi
                # flip back
                x[:, -1] *= -1
                x[:, 1] *= -1

        return log_amplitude_ratio

    def amplitude_ratio(self, x):
        return self.log_amplitude_ratio(x).exp()


class ComplexBaseModel(nn.Module):
    """
    one has to implement forward(), log_prob_and_phi() and sample() method
    """
    def forward(self):
        pass

    def log_prob_and_phi(self):
        pass

    def sample(self):
        pass

    def log_psi(self, x):
        """log Psi(s) = 1/2 log P(s) + i * phi(s)

        Args:
            x (Tensor): quantum state, (batch_size, L), values in {-1, +1}

        Returns:
            Tensor: log amplitude, (batch_size, )
        """
        log_prob, phi = self.log_prob_and_phi(x)
        return log_prob / 2 + 1j * phi

    def local_energy(self, x, ham):
        """for a given state x, find its non-zero matrix elements <x|H|x`> 
        and the coresponding amplitude ratio Psi(x)/Psi(x`) and then calculate the local energy E_loc

        Args:
            x (Tensor): quantum state, (batch_size, L) or (batch_size, L, L), values in {-1, +1}
            ham (Hamiltonian): specific hamiltonian

        Returns:
            Tensor: local energy E_loc, (batch_size, )
        """
        matrix_elements = ham.find_matrix_elements(x)
        amplitude_ratio = self.amplitude_ratio(x)
        E_loc = torch.sum(matrix_elements * amplitude_ratio, dim=1)

        return E_loc

    def log_amplitude_ratio(self, x):
        """for a given state x, calculate its log amplitude ratio log psi(x`) - log psi(x)

        Args:
            x (Tensor): quantum state, (batch_size, L) or (batch_size, L, L), values in {-1, +1}

        Returns:
            Tensor: log amplitude ratio, (batch_size, *)
        """
        if self.ham == 'ising':
            if self.dim == 1:
                log_amplitude_ratio = torch.zeros((self.batch_size, self.L + 1), dtype=torch.cfloat, device=self.device)
                log_psi = self.log_psi(x)
                for i in range(self.L):
                    # flip spin
                    x[:, i] *= -1
                    log_psi_prime = self.log_psi(x)
                    log_amplitude_ratio[:, i + 1] = log_psi_prime - log_psi
                    # flip back
                    x[:, i] *= -1
            elif self.dim == 2:
                log_amplitude_ratio = torch.zeros((self.batch_size, self.L**2 + 1),
                                                  dtype=torch.cfloat,
                                                  device=self.device)
                log_psi = self.log_psi(x)
                for i in range(self.L):
                    for j in range(self.L):
                        # flip spin
                        x[:, i, j] *= -1
                        log_psi_prime = self.log_psi(x)
                        log_amplitude_ratio[:, i * self.L + j + 1] = log_psi_prime - log_psi
                        # flip back
                        x[:, i, j] *= -1
        elif self.ham == 'j1j2':
            log_amplitude_ratio = -torch.zeros(
                (self.batch_size, self.L * 2 + 1), dtype=torch.cfloat, device=self.device)
            log_psi = self.log_psi(x)
            # J1 part
            for i in range(self.L - 1):
                # flip two spins (nearest)
                x[:, i] *= -1
                x[:, i + 1] *= -1
                log_psi_prime = self.log_psi(x)
                log_amplitude_ratio[:, i + 1] = log_psi_prime - log_psi
                # flip back
                x[:, i] *= -1
                x[:, i + 1] *= -1
            if self.pbc:
                # flip two spins (nearest)
                x[:, -1] *= -1
                x[:, 0] *= -1
                log_psi_prime = self.log_psi(x)
                log_amplitude_ratio[:, self.L] = log_psi_prime - log_psi
                # flip back
                x[:, -1] *= -1
                x[:, 0] *= -1
            # J2 part
            for i in range(self.L - 2):
                # flip two spins (next nearest)
                x[:, i] *= -1
                x[:, i + 2] *= -1
                log_psi_prime = self.log_psi(x)
                log_amplitude_ratio[:, i + self.L + 1] = log_psi_prime - log_psi
                # flip back
                x[:, i] *= -1
                x[:, i + 2] *= -1
            if self.pbc:
                # flip two spins (next nearest)
                x[:, -2] *= -1
                x[:, 0] *= -1
                log_psi_prime = self.log_psi(x)
                log_amplitude_ratio[:, -2] = log_psi_prime - log_psi
                # flip back
                x[:, -2] *= -1
                x[:, 0] *= -1
                # flip two spins (next nearest)
                x[:, -1] *= -1
                x[:, 1] *= -1
                log_psi_prime = self.log_psi(x)
                log_amplitude_ratio[:, -1] = log_psi_prime - log_psi
                # flip back
                x[:, -1] *= -1
                x[:, 1] *= -1

        return log_amplitude_ratio

    def amplitude_ratio(self, x):
        return self.log_amplitude_ratio(x).exp()
