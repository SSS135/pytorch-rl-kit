import torch
from torch import nn as nn
from torch.nn import functional as F


class CategoricalValueEncoder(nn.Module):
    def __init__(self, approx_max, num_bins, eps=1e-6):
        super().__init__()
        self.num_bins = num_bins
        self.approx_max = approx_max
        self.eps = eps

    def forward(self, bins):
        limit = self.approx_max ** 0.5
        lin = torch.linspace(-limit, limit, bins.shape[-1], device=bins.device, dtype=bins.dtype)
        value = (bins.detach().softmax(-1) * self._linear_to_scaled(lin)).sum(-1)
        assert value.shape == bins.shape[:-1] and bins.shape[-1] == self.num_bins
        return value

    def logp(self, bins, value):
        assert bins.shape[:-1] == value.shape and bins.shape[-1] == self.num_bins
        D = bins.shape[-1]
        limit = self.approx_max ** 0.5
        value = self._scaled_to_linear(value.detach()) / (2 * limit) + 0.5
        index_f = value.mul(D - 1).clamp(self.eps, D - 1 - self.eps)
        low_idx = index_f.floor()
        low_idx_bin = low_idx.unsqueeze(-1).long()
        frac = index_f - low_idx
        logp = F.log_softmax(bins, dim=-1)
        logp_low = logp.gather(dim=-1, index=low_idx_bin).squeeze(-1)
        logp_high = logp.gather(-1, index=low_idx_bin + 1).squeeze(-1)
        logp = torch.lerp(logp_low, logp_high, frac)
        assert logp.shape == bins.shape[:-1], (logp.shape, bins.shape)
        return logp

    def _scaled_to_linear(self, x):
        return x.sign() * ((x.abs() + 1).sqrt() - 1 + self.eps * x.abs())

    def _linear_to_scaled(self, a):
        sign = a.sign()
        a = a.abs().double()
        eps = self.eps
        x = sign * (2 * (a + 1) * eps - (4 * eps * (a + eps + 1) + 1).sqrt() + 1) / (2 * eps ** 2)
        return x.float()