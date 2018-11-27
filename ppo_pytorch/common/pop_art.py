from typing import Tuple

import torch


class PopArt:
    def __init__(self, beta=0.99, eps=(1e-4, 1e6)):
        assert 0 <= beta < 1
        assert 0 <= eps[0] < eps[1]
        self.beta = beta
        self.eps = eps
        self._mean = 0
        self._square = 0
        self._update_iteration = 0

    def update_statistics(self, returns: torch.Tensor) -> Tuple[float, float]:
        self._update_iteration += 1
        bias_corr = 1 - self.beta ** self._update_iteration
        self._mean = self.beta * self._mean + (1 - self.beta) * returns.mean().item()
        self._square = self.beta * self._square + (1 - self.beta) * returns.pow(2).mean().item()
        mean = self._mean / bias_corr
        std = max(self.eps[0], min(self.eps[1], (self._square - self._mean ** 2) ** 0.5 / bias_corr))
        return mean, std