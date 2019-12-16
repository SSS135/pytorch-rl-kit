import random

import torch
import torch.nn as nn


class ActivationNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = 0

    def forward(self, input: torch.Tensor, always_run=False):
        if not always_run and not input.requires_grad:
            return input
        assert input.dim() == 2 or input.dim() == 4
        B, C = input.shape[:2]
        x = input.reshape(B, C, -1)
        std, mean = torch.std_mean(x, dim=(0, 2))
        self.loss = 0.5 * ((std - 1).pow(2).mean() + mean.pow(2).mean())
        # self.loss = x.pow(2).mean(dim=2).sqrt().sub(1).pow(2).mean().mul(0.5)
        return input


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
        if isinstance(m, ActivationNorm) and torch.is_tensor(m.loss):
            losses.append(m.loss.view(1))
            m.loss = 0
    return torch.cat(losses).mean() if len(losses) > 0 else torch.scalar_tensor(0.0)
