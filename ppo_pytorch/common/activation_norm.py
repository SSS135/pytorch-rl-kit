import random
from typing import Optional

import torch
import torch.nn as nn


class ActivationNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self._input = None

    def forward(self, input: torch.Tensor, always_run=False):
        if not always_run and not input.requires_grad:
            return input
        assert input.dim() == 2 #or input.dim() == 4
        self._input = input
        return input

    def get_loss(self, clear=True) -> Optional[torch.Tensor]:
        if self._input is None:
            return None
        B, C = self._input.shape[:2]
        x = self._input.reshape(B, C, -1)
        std, mean = torch.std_mean(x, dim=0)
        loss = 0.5 * (std.pow(2).mean() + mean.pow(2).mean() - 2 * std.log().mean())
        if clear:
            self._input = None
        return loss


class ActivationNormWrapper(nn.Module):
    def __init__(self, wrapped_module):
        super().__init__()
        self.wrapped_module = wrapped_module
        self.act_norm = ActivationNorm()

    def forward(self, input: torch.Tensor):
        output = self.wrapped_module(input)
        if input.requires_grad:
            self.act_norm(self.wrapped_module(input.detach()), always_run=True)
        return output


def activation_norm_loss(module):
    losses = []
    for m in module.modules():
        if isinstance(m, ActivationNorm):
            loss = m.get_loss()
            if loss is not None:
                losses.append(loss.view(1))
    return torch.cat(losses).mean() if len(losses) > 0 else torch.scalar_tensor(0.0)
