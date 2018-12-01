import numpy as np


class OnlineNormalizer:
    def __init__(self, eps=(1e-3, 1e5), absmax=5, scale=True, center=True, single_value=True):
        self.eps = eps
        self.absmax = absmax
        self.scale = scale
        self.center = center
        self.single_value = single_value
        self._iter = 0
        self._mean = 0.0
        self._M2 = 0.0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self._iter += 1
        delta_old = x - self._mean
        self._mean = self._mean + delta_old / self._iter if self.center else 0
        delta_new = x - self._mean
        self._M2 += delta_old * delta_new

        var = self._M2 / self._iter if self._iter >= 2 else 1
        std = np.sqrt(var).clip(self.eps[0], self.eps[1]) if self.scale else 1
        mean = self._mean if self._iter >= 2 else 0
        if self.single_value:
            mean = np.mean(mean)
            std = np.mean(std)
        x = x - mean if self.center else x
        x = x / std if self.scale else x
        return x.clip(-self.absmax, self.absmax)
