import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler


class ValueDecay:
    def __init__(self, start_value, end_value, end_epoch, temp=4, exp=True, eps=1e-7):
        """
        Decays some value from `start_value` to `end_value` for `end_epoch` epochs.
            Supports linear and exponential decay.
        Args:
            start_value: Initial value
            end_value: Final value
            end_epoch: Final epoch
            temp: Slope steepness for exponential decay
            exp: Use exponential decay
            eps: Used for numerical stability
        """
        self.start_value = start_value
        self.end_value = end_value
        self.temp = temp
        self.end_epoch = end_epoch
        self.exp = exp
        self.eps = eps
        self._epoch = 0

    def step(self, epoch=None):
        if epoch is None:
            epoch = self._epoch + 1
        self._epoch = np.clip(epoch, 0, self.end_epoch)

    @property
    def value(self):
        t = self._epoch / self.end_epoch
        if self.exp:
            return self._explerp(self.start_value, self.end_value, t)
        else:
            return self._linlerp(self.start_value, self.end_value, t)

    def _linlerp(self, a, b, t):
        return a + (b - a) * t

    def _explerp(self, a, b, t):
        temp = self.temp + self.eps
        if b < a:
            a, b = b, a
            t = 1 - t
        t = (np.exp(t * temp) - 1) / (np.exp(temp) - 1)
        return a + (b - a) * t.item()

    def __repr__(self):
        return f'ValueDecay(start_value={self.start_value}, end_value={self.end_value}, end_epoch={self.end_epoch}, ' \
               f'temp={self.temp}, exp={self.exp}, eps={self.eps})'

    def __str__(self):
        return self.__repr__()


class DecayLR(_LRScheduler):
    def __init__(self, optimizer: optim.Optimizer, *args, **kwargs):
        """
        Pytorch learning rate scheduler based on `ValueDecay`
        Args:
            optimizer: Model optimizer
            *args: Passed to `ValueDecay` constructor
            **kwargs: Passed to `ValueDecay` constructor
        """
        self.value_decay = ValueDecay(*args, **kwargs)
        super().__init__(optimizer, -1)

    def get_lr(self):
        self.value_decay.step(self.last_epoch)
        value = self.value_decay.value
        return [base_lr * value for base_lr in self.base_lrs]