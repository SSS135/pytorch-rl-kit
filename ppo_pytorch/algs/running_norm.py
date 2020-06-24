from torch import nn
from torch import Tensor


class RunningNorm:
    def __init__(self, momentum=0.99, mean_norm=True, eps=1e-8):
        self.momentum = momentum
        self.mean_norm = mean_norm
        self.eps = eps
        self._mean = 0
        self._square = 0
        self._iter = 0

    def __call__(self, values, update_stats=True):
        if update_stats:
            self._mean = self.momentum * self._mean + (1 - self.momentum) * values.mean().item()
            self._square = self.momentum * self._square + (1 - self.momentum) * values.pow(2).mean().item()
            self._iter += 1

        bias_corr = 1 - self.momentum ** self._iter
        mean = self._mean / bias_corr
        square = self._square / bias_corr

        if self.mean_norm:
            std = (square - mean ** 2) ** 0.5
            values = (values - mean) / max(std, self.eps)
        else:
            rms = square ** 0.5
            values = values / max(rms, self.eps)

        return values


class GradRunningNorm(nn.Module):
    def __init__(self, weight=1.0, momentum=0.99, eps=1e-8):
        super().__init__()
        self.weight = weight
        self._norm = RunningNorm(momentum, mean_norm=False, eps=eps)
        self.register_backward_hook(self.backward_hook)

    def backward_hook(self, module, grad_input, grad_output):
        grad_input, = grad_input
        grad_input = self._norm(grad_input)
        return self.weight * grad_input,

    def forward(self, input: Tensor):
        return input.clone()