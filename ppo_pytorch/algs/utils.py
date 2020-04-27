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
def v_mpo_loss(kl: torch.Tensor, logp: torch.Tensor, advantages: torch.Tensor, advantages_upgo: torch.Tensor,
                       vtrace_p: torch.Tensor, kl_pull: float) \
        -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    assert kl.dim() == logp.dim() == advantages.dim() == 1

    advantages = advantages * vtrace_p + advantages_upgo

    mask = vtrace_p > 0.1
    if mask.float().mean().item() < 0.1:
        return None

    adv_masked = advantages[mask]
    softmax = adv_masked.softmax(0)

    loss_policy = softmax.sub(softmax.median()).mul_(softmax.numel()).mul_(vtrace_p[mask]).clamp_(-5, 5).detach_().mul_(-logp[mask])
    loss_kl = kl_pull * kl

    assert loss_policy.shape == adv_masked.shape, (loss_policy.shape, adv_masked.shape)
    assert loss_kl.shape == kl.shape, (loss_kl.shape, kl.shape)

    return loss_policy.mean(), loss_kl.mean()


@torch.jit.script
def scaled_impala_loss(kl_target: torch.Tensor, logp: torch.Tensor, advantages: torch.Tensor, advantages_upgo: torch.Tensor,
                       vtrace_p: torch.Tensor, kl_pull: float, kl_limit: float) \
        -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    assert advantages.shape == vtrace_p.shape == advantages_upgo.shape and advantages.dim() == 1
    assert kl_target.shape == logp.shape and kl_target.dim() == 2

    advantages = advantages.mul(vtrace_p).add_(advantages_upgo)

    kl_mask = (kl_target <= kl_limit).float()
    loss_policy = advantages.clamp(-5, 5).unsqueeze_(-1).detach_().mul(-logp).mul_(kl_mask)
    loss_kl = kl_pull * kl_target

    assert loss_policy.shape[:-1] == advantages.shape, (loss_policy.shape, advantages.shape)
    assert loss_kl.shape == kl_target.shape, (loss_kl.shape, kl_target.shape)

    return loss_policy.mean(), loss_kl.mean()


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