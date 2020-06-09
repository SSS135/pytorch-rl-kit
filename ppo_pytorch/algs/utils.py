from typing import Optional, Tuple

import torch
import torch.jit
from torch import nn


def lerp_module_(start, end, factor):
    if isinstance(start, nn.Module):
        start = start.state_dict().values()
    if isinstance(end, nn.Module):
        end = end.state_dict().values()

    for a, b in zip(start, end):
        if not torch.is_floating_point(a):
            a.data.copy_(b.data)
        else:
            a.data.lerp_(b.data.to(a.device), factor)


@torch.jit.script
def v_mpo_loss(logp: torch.Tensor, advantages: torch.Tensor, kl_target: torch.Tensor, kl_limit: float) \
        -> Optional[torch.Tensor]:
    assert kl_target.shape == logp.shape and kl_target.dim() == 2

    mask = kl_target.mean(-1) <= kl_limit
    if mask.float().mean().item() < 0.2:
        return None

    softmax = advantages[mask].softmax(0)
    softmax = softmax.sub(softmax.median()).mul_(softmax.numel())
    logp_masked = logp[mask.unsqueeze(-1).expand_as(logp)].view(-1, logp.shape[1])
    loss_policy = softmax.detach().clamp(-5, 5).unsqueeze(-1).mul(-logp_masked)

    assert loss_policy.ndim == 2, loss_policy.shape

    return loss_policy.mean()


@torch.jit.script
def impala_loss(logp: torch.Tensor, advantages: torch.Tensor, kl_target: torch.Tensor, kl_limit: float) -> torch.Tensor:
    assert advantages.dim() == 1
    assert kl_target.shape == logp.shape and kl_target.dim() == 2

    kl_mask = (kl_target <= kl_limit).float()
    loss_policy = advantages.clamp(-5, 5).unsqueeze_(-1).detach_().mul(-logp).mul_(kl_mask)

    assert loss_policy.shape[:-1] == advantages.shape, (loss_policy.shape, advantages.shape)

    return loss_policy.mean()


class RunningNorm:
    def __init__(self, momentum=0.99, mean_norm=True):
        self.momentum = momentum
        self.mean_norm = mean_norm
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
            values = (values - mean) / max(std, 1e-5)
        else:
            rms = square ** 0.5
            values = values / max(rms, 1e-5)

        return values