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


@torch.compile
def v_mpo_loss(logp: Tensor, advantages: Tensor, adv_temp: Tensor, target_temp: float) -> Tensor:
    assert advantages.shape == logp.shape[:-1]

    mask = advantages >= advantages.median()
    advantages = advantages[mask]
    logp = logp[mask.unsqueeze(-1).expand_as(logp)].view(-1, logp.shape[1])

    loss_policy = advantages.div(adv_temp.detach()).softmax(0).unsqueeze(-1).mul(-logp)
    loss_temp = adv_temp * target_temp + adv_temp * (torch.logsumexp(advantages / adv_temp, 0) - math.log(advantages.shape[0]))

    assert loss_policy.ndim == 2, loss_policy.shape
    assert loss_temp.ndim == 0

    return loss_policy.mean(-1).sum() + loss_temp.mean()


@torch.compile
def impala_loss(logp: Tensor, advantages: Tensor) -> Tensor:
    assert advantages.shape == logp.shape[:-1]
    loss_policy = advantages.unsqueeze(-1).detach().mul(-logp)
    assert loss_policy.shape[:-1] == advantages.shape, (loss_policy.shape, advantages.shape)
    return loss_policy.mean()


