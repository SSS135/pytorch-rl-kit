import random

import torch
import torch.nn as nn


class ActivationNorm(nn.Module):
    def __init__(self, num_channels, num_groups, num_rand_sets=100):
        super().__init__()
        self.num_channels = num_channels
        self.num_groups = num_groups
        self.num_rand_sets = num_rand_sets
        assert num_channels % num_groups == 0
        self.register_buffer('rand_indexes', torch.randperm(num_rand_sets * num_channels).fmod_(num_channels).reshape(num_rand_sets, num_channels))
        self.loss = 0

    def forward(self, input: torch.Tensor):
        if not input.requires_grad:
            return input
        assert input.dim() == 2 or input.dim() == 4
        B, C = input.shape[:2]
        x = input.reshape(B, C, -1)
        # D = x.shape[2]
        # indexes = self.rand_indexes[random.randrange(self.num_rand_sets)]
        # x = x.index_select(1, indexes)
        # x = x.reshape(B, self.num_groups, C // self.num_groups * D)
        std, mean = torch.std_mean(x, dim=(0, 2))
        self.loss = 0.5 * ((std - 1).pow(2).mean() + mean.pow(2).mean())
        # self.loss = x.pow(2).mean(dim=2).sqrt().sub(1).pow(2).mean().mul(0.5)
        return input


def activation_norm_loss(module):
    losses = []
    for m in module.modules():
        if isinstance(m, ActivationNorm) and torch.is_tensor(m.loss):
            losses.append(m.loss.view(1))
            m.loss = 0
    return torch.cat(losses).mean() if len(losses) > 0 else torch.scalar_tensor(0.0)
