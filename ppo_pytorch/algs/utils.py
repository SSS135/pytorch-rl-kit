import math

import torch
import torch.jit
from torch import nn
from torch import Tensor


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
def v_mpo_loss(logp: Tensor, advantages: Tensor, kl_target: Tensor, kl_limit: float,
               adv_temp: Tensor, target_temp: float) -> Tensor:
    assert kl_target.shape == logp.shape and kl_target.dim() == 2
    assert advantages.shape == kl_target.shape[:1]

    mask = advantages >= advantages.median()
    advantages = advantages[mask]
    logp = logp[mask.unsqueeze(-1).expand_as(logp)].view(-1, logp.shape[1])
    kl_target = kl_target[mask.unsqueeze(-1).expand_as(kl_target)].view(-1, kl_target.shape[1])
    kl_mask = (kl_target <= kl_limit).float()

    loss_policy = advantages.div(adv_temp.detach()).softmax(0).unsqueeze(-1).mul(-logp).mul(kl_mask)
    loss_temp = adv_temp * target_temp + adv_temp * (torch.logsumexp(advantages / adv_temp, 0) - math.log(advantages.shape[0]))

    assert loss_policy.ndim == 2, loss_policy.shape
    assert loss_temp.ndim == 0

    return loss_policy.mean(-1).sum() + loss_temp.mean()


@torch.jit.script
def impala_loss(logp: Tensor, advantages: Tensor, kl_target: Tensor, kl_limit: float) -> Tensor:
    assert advantages.dim() == 1
    assert kl_target.shape == logp.shape and kl_target.dim() == 2

    kl_mask = (kl_target <= kl_limit).float()
    loss_policy = advantages.clamp(-5, 5).unsqueeze_(-1).detach_().mul(-logp).mul_(kl_mask)

    assert loss_policy.shape[:-1] == advantages.shape, (loss_policy.shape, advantages.shape)

    return loss_policy.mean()


