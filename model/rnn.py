import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ComplexBaseModel, RealBaseModel


class PositiveRNN(RealBaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.L = kwargs['L']
        self.hidden_size = kwargs['hidden_size']
        self.device = kwargs['device']
        self.batch_size = kwargs['batch_size']
        self.ham = kwargs['ham']
        self.dim = kwargs['dim']
        self.pbc = kwargs['pbc']

        # build positive RNN wave function
        self.rnn = nn.GRUCell(2, self.hidden_size)
        self.fc1 = nn.Linear(self.hidden_size, 2)

    def forward(self, x, h):
        """

        Args:
            x (Tensor): quantum state of sigma_i (i = 0 ... L-1) (batch_size, ), values in {-1, +1}

        Returns:
            Tensor: h_i+1 (batch_size, hidden) and y_i+1 (batch_size, 2)
        """
        embedded_x = torch.stack([(x + 1) / 2, 1.0 - (x + 1) / 2], dim=1)  # (batch_size, 2)
        h_next = self.rnn(embedded_x, h)
        y = F.log_softmax(self.fc1(h_next), dim=1)
        return h_next, y

    def log_prob(self, x):
        """

        Args:
            x (Tensor): quantum state, (batch_size, L) or (batch_size, L, L), values in {-1, +1}

        Returns:
            Tensor: log probability, (batch_size, )
        """
        batch_size = x.shape[0]
        x = x.view(batch_size, -1) if self.dim == 2 else x  # Z order reshape for 2D model
        log_prob = torch.zeros_like(x)
        mask = (1 + x) / 2
        # use fixed sigma_0 and h_0 to compute h_1 and y_1
        x_init = torch.zeros(batch_size, dtype=torch.float, device=self.device)
        h_init = torch.zeros(batch_size, self.hidden_size, dtype=torch.float, device=self.device)
        h, y = self.forward(x_init, h_init)
        log_prob[:, 0] = y[:, 0] * mask[:, 0] + y[:, 1] * (1.0 - mask[:, 0])

        for i in range(1, self.L if self.dim == 1 else self.L**2):
            h, y = self.forward(x[:, i - 1], h)
            log_prob[:, i] = y[:, 0] * mask[:, i] + y[:, 1] * (1.0 - mask[:, i])

        return log_prob.sum(dim=1)

    def sample(self):
        """

        Returns:
            Tensor: sampled quantum state, (batch_size, L) or (batch_size, L, L), values in {-1, +1}
        """
        samples = torch.empty((self.batch_size, self.L if self.dim == 1 else self.L**2),
                              dtype=torch.float,
                              device=self.device)

        # use fixed sigma_0 and h_0 to compute h_1 and y_1, and then sample from y_1
        x_init = torch.zeros(self.batch_size, dtype=torch.float, device=self.device)
        h_init = torch.zeros(self.batch_size, self.hidden_size, dtype=torch.float, device=self.device)
        h, y = self.forward(x_init, h_init)
        p = torch.exp(y)[:, 0]
        samples[:, 0] = torch.bernoulli(p) * 2 - 1

        for i in range(1, self.L if self.dim == 1 else self.L**2):
            h, y = self.forward(samples[:, i - 1], h)
            p = torch.exp(y)[:, 0]
            samples[:, i] = torch.bernoulli(p) * 2 - 1

        return samples if self.dim == 1 else samples.view(self.batch_size, self.L, self.L)


class ComplexRNN(ComplexBaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.L = kwargs['L']
        self.hidden_size = kwargs['hidden_size']
        self.device = kwargs['device']
        self.batch_size = kwargs['batch_size']
        self.ham = kwargs['ham']
        self.dim = kwargs['dim']
        self.pbc = kwargs['pbc']

        # build complex RNN wave function
        self.rnn = nn.GRUCell(2, self.hidden_size)
        self.fc1 = nn.Linear(self.hidden_size, 2)
        self.fc2 = nn.Linear(self.hidden_size, 2)

    def forward(self, x, h):
        """

        Args:
            x (Tensor): quantum state of sigma_i (i = 0 ... L-1) (batch_size, ), values in {-1, +1}

        Returns:
            Tensor: h_i+1 (batch_size, hidden), y^1_i+1 (batch_size, 2) and y^2_i+1 (batch_size, 2)
        """
        embedded_x = torch.stack([(x + 1) / 2, 1.0 - (x + 1) / 2], dim=1)  # (batch_size, 2)
        h_next = self.rnn(embedded_x, h)
        y1 = F.log_softmax(self.fc1(h_next), dim=1)
        y2 = math.pi * F.softsign(self.fc2(h_next))
        return h_next, y1, y2

    def log_prob_and_phi(self, x):
        """

        Args:
            x (Tensor): quantum state, (batch_size, L) or (batch_size, L, L), values in {-1, +1}

        Returns:
            Tensor: log probability (batch_size, ) and phase (batch_size, )
        """
        batch_size = x.shape[0]
        x = x.view(batch_size, -1) if self.dim == 2 else x  # Z order reshape for 2D model
        log_prob = torch.zeros_like(x)
        phi = torch.zeros_like(x)
        mask = (1 + x) / 2
        # use fixed sigma_0 and h_0 to compute h_1 and y_1
        x_init = torch.zeros(batch_size, dtype=torch.float, device=self.device)
        h_init = torch.zeros(batch_size, self.hidden_size, dtype=torch.float, device=self.device)
        h, y1, y2 = self.forward(x_init, h_init)
        log_prob[:, 0] = y1[:, 0] * mask[:, 0] + y1[:, 1] * (1.0 - mask[:, 0])
        phi[:, 0] = y2[:, 0] * mask[:, 0] + y2[:, 1] * (1.0 - mask[:, 0])

        for i in range(1, self.L if self.dim == 1 else self.L**2):
            h, y1, y2 = self.forward(x[:, i - 1], h)
            log_prob[:, i] = y1[:, 0] * mask[:, i] + y1[:, 1] * (1.0 - mask[:, i])
            phi[:, i] = y2[:, 0] * mask[:, i] + y2[:, 1] * (1.0 - mask[:, i])

        return log_prob.sum(dim=1), phi.sum(dim=1)

    def sample(self):
        """

        Returns:
            Tensor: sampled quantum state, (batch_size, L) or (batch_size, L, L), values in {-1, +1}
        """
        samples = torch.empty((self.batch_size, self.L if self.dim == 1 else self.L**2),
                              dtype=torch.float,
                              device=self.device)

        # use fixed sigma_0 and h_0 to compute h_1 and y_1, and then sample from y_1
        x_init = torch.zeros(self.batch_size, dtype=torch.float, device=self.device)
        h_init = torch.zeros(self.batch_size, self.hidden_size, dtype=torch.float, device=self.device)
        h, y, _ = self.forward(x_init, h_init)
        p = torch.exp(y)[:, 0]
        samples[:, 0] = torch.bernoulli(p) * 2 - 1

        for i in range(1, self.L if self.dim == 1 else self.L**2):
            h, y, _ = self.forward(samples[:, i - 1], h)
            p = torch.exp(y)[:, 0]
            samples[:, i] = torch.bernoulli(p) * 2 - 1

        return samples if self.dim == 1 else samples.view(self.batch_size, self.L, self.L)
