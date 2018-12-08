from typing import Tuple

import torch


def clip(x, x_min, x_max):
    return max(x_min, min(x_max, x))


class PopArt:
    def __init__(self, beta=0.95, use_rms=False, eps=(1e-4, 1e6)):
        assert 0 <= beta < 1
        assert 0 <= eps[0] < eps[1]
        self.beta = beta
        self.eps = eps
        self.use_rms = use_rms
        self._mean = 0
        self._square = 0
        self._update_iteration = 0

    @property
    def statistics(self):
        if self._update_iteration == 0:
            return 0, 1
        else:
            bias_corr = 1 - self.beta ** self._update_iteration
            square = self._square / bias_corr
            mean = self._mean / bias_corr
            if self.use_rms:
                rms = clip(square ** 0.5, *self.eps)
                return 0.0, rms
            else:
                std = clip((square - mean ** 2) ** 0.5, *self.eps)
                return mean, std

    def update_statistics(self, returns: torch.Tensor):
        self._update_iteration += 1
        self._square = self.beta * self._square + (1 - self.beta) * returns.pow(2).mean().item()
        self._mean = self.beta * self._mean + (1 - self.beta) * returns.mean().item()