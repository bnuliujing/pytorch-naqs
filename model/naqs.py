import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ComplexBaseModel, RealBaseModel


class MaskedConv1d(nn.Conv1d):
    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, k = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, k // 2 + (mask_type == 'B'):] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class PositiveNAQS(RealBaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.L = kwargs['L']
        self.net_depth = kwargs['net_depth']
        self.hidden_size = kwargs['hidden_size']
        self.kernel_size = kwargs['kernel_size']
        self.padding = kwargs['padding']
        self.device = kwargs['device']
        self.batch_size = kwargs['batch_size']
        self.ham = kwargs['ham']
        self.dim = kwargs['dim']
        self.pbc = kwargs['pbc']

        assert self.kernel_size == 2 * self.padding + 1
        assert self.net_depth >= 2

        # build NQS model
        net = []

        if self.dim == 1:
            net.append(MaskedConv1d('A', 1, self.hidden_size, self.kernel_size, padding=self.padding))
            net.append(nn.ReLU())

            for _ in range(self.net_depth - 1):
                net.append(MaskedConv1d('B', self.hidden_size, self.hidden_size, self.kernel_size,
                                        padding=self.padding))
                net.append(nn.ReLU())

            net.append(nn.Conv1d(self.hidden_size, 2, 1))
            net.append(nn.LogSoftmax(dim=1))

            self.conv = nn.Sequential(*net)

        elif self.dim == 2:
            net.append(MaskedConv2d('A', 1, self.hidden_size, self.kernel_size, padding=self.padding))
            net.append(nn.ReLU())

            for _ in range(self.net_depth - 1):
                net.append(MaskedConv2d('B', self.hidden_size, self.hidden_size, self.kernel_size,
                                        padding=self.padding))
                net.append(nn.ReLU())

            net.append(nn.Conv2d(self.hidden_size, 2, 1))
            net.append(nn.LogSoftmax(dim=1))

            self.conv = nn.Sequential(*net)

    def forward(self, x):
        """

        Args:
            x (Tensor): quantum state, (batch_size, L) or (batch_size, L, L), values in {-1, +1}

        Returns:
            Tensor: (batch_size, 2, L) or (batch_size, 2, L, L)
        """
        return self.conv(x.unsqueeze(1))

    def log_prob(self, x):
        """

        Args:
            x (Tensor): quantum state, (batch_size, L) or (batch_size, L, L), values in {-1, +1}

        Returns:
            Tensor: log probability, (batch_size, )
        """
        mask = (1 + x) / 2
        y = self.forward(x)
        if self.dim == 1:
            log_prob = y[:, 0, :] * mask + y[:, 1, :] * (1.0 - mask)
        elif self.dim == 2:
            log_prob = y[:, 0, :, :] * mask + y[:, 1, :, :] * (1.0 - mask)

        return log_prob.sum(dim=1) if self.dim == 1 else log_prob.sum(dim=(1, 2))

    def sample(self):
        """

        Returns:
            Tensor: sampled quantum state, (batch_size, L) or (batch_size, L, L), values in {-1, +1}
        """
        size = (self.batch_size, self.L) if self.dim == 1 else (self.batch_size, self.L, self.L)
        samples = torch.randint(0, 2, size=size, dtype=torch.float, device=self.device) * 2 - 1
        if self.dim == 1:
            for idx in range(self.L):
                p = torch.exp(self.forward(samples)[:, :, idx])[:, 0]
                samples[:, idx] = torch.bernoulli(p) * 2 - 1
        elif self.dim == 2:
            for i in range(self.L):
                for j in range(self.L):
                    p = torch.exp(self.forward(samples)[:, :, i, j])[:, 0]
                    samples[:, i, j] = torch.bernoulli(p) * 2 - 1

        return samples


class ComplexNAQS(ComplexBaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.L = kwargs['L']
        self.net_depth = kwargs['net_depth']
        self.hidden_size = kwargs['hidden_size']
        self.kernel_size = kwargs['kernel_size']
        self.padding = kwargs['padding']
        self.device = kwargs['device']
        self.batch_size = kwargs['batch_size']
        self.ham = kwargs['ham']
        self.dim = kwargs['dim']
        self.pbc = kwargs['pbc']

        assert self.kernel_size == 2 * self.padding + 1
        assert self.net_depth >= 2

        # build NQS model
        net = []

        if self.dim == 1:
            net.append(MaskedConv1d('A', 1, self.hidden_size, self.kernel_size, padding=self.padding))
            net.append(nn.ReLU())

            for _ in range(self.net_depth - 1):
                net.append(MaskedConv1d('B', self.hidden_size, self.hidden_size, self.kernel_size,
                                        padding=self.padding))
                net.append(nn.ReLU())

            self.conv = nn.Sequential(*net)

            self.fc1 = nn.Conv1d(self.hidden_size, 2, 1)
            self.fc2 = nn.Conv1d(self.hidden_size, 2, 1)

        elif self.dim == 2:
            net.append(MaskedConv2d('A', 1, self.hidden_size, self.kernel_size, padding=self.padding))
            net.append(nn.ReLU())

            for _ in range(self.net_depth - 1):
                net.append(MaskedConv2d('B', self.hidden_size, self.hidden_size, self.kernel_size,
                                        padding=self.padding))
                net.append(nn.ReLU())

            self.conv = nn.Sequential(*net)

            self.fc1 = nn.Conv2d(self.hidden_size, 2, 1)
            self.fc2 = nn.Conv2d(self.hidden_size, 2, 1)

    def forward(self, x):
        """

        Args:
            x (Tensor): quantum state, (batch_size, L) or (batch_size, L, L), values in {-1, +1}

        Returns:
            Tensor: (batch_size, 2, L) or (batch_size, 2, L, L)
        """
        x = self.conv(x.unsqueeze(1))  # (batch_size, 2, L) or (batch_size, 2, L, L)
        y1 = F.log_softmax(self.fc1(x), dim=1)
        y2 = math.pi * F.softsign(self.fc2(x))
        return y1, y2

    def log_prob_and_phi(self, x):
        """

        Args:
            x (Tensor): quantum state, (batch_size, L) or (batch_size, L, L), values in {-1, +1}

        Returns:
            Tensor: log probability (batch_size, ) and phase (batch_size, )
        """
        mask = (1 + x) / 2
        y1, y2 = self.forward(x)
        if self.dim == 1:
            log_prob = y1[:, 0, :] * mask + y1[:, 1, :] * (1.0 - mask)
            phi = y2[:, 0, :] * mask + y2[:, 1, :] * (1.0 - mask)
            log_prob = log_prob.sum(dim=1)
            phi = phi.sum(dim=1)
        elif self.dim == 2:
            log_prob = y1[:, 0, :, :] * mask + y1[:, 1, :, :] * (1.0 - mask)
            phi = y2[:, 0, :, :] * mask + y2[:, 1, :, :] * (1.0 - mask)
            log_prob = log_prob.sum(dim=(1, 2))
            phi = phi.sum(dim=(1, 2))

        return log_prob, phi

    def sample(self):
        """

        Returns:
            Tensor: sampled quantum state, (batch_size, L) or (batch_size, L, L), values in {-1, +1}
        """
        size = (self.batch_size, self.L) if self.dim == 1 else (self.batch_size, self.L, self.L)
        samples = torch.randint(0, 2, size=size, dtype=torch.float, device=self.device) * 2 - 1
        if self.dim == 1:
            for i in range(self.L):
                log_p, _ = self.forward(samples)
                p = torch.exp(log_p[:, :, i])[:, 0]
                samples[:, i] = torch.bernoulli(p) * 2 - 1
        elif self.dim == 2:
            for i in range(self.L):
                for j in range(self.L):
                    log_p, _ = self.forward(samples)
                    p = torch.exp(log_p[:, :, i, j])[:, 0]
                    samples[:, i, j] = torch.bernoulli(p) * 2 - 1

        return samples
