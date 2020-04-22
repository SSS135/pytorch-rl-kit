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
def v_mpo_loss(kl: torch.Tensor, kl_replay: torch.Tensor, logp: torch.Tensor, advantages: torch.Tensor, advantages_upgo: torch.Tensor,
                       vtrace_p: torch.Tensor, nu: torch.Tensor, alpha: torch.Tensor,
                       eps_nu: float, eps_alpha: float) \
        -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    assert kl.dim() == logp.dim() == advantages.dim() == 1
    assert nu.shape == alpha.shape == ()
    assert eps_nu < 1, eps_nu

    advantages = advantages * vtrace_p + advantages_upgo

    mask = vtrace_p > 0.1
    # if mask.sum().item() == 0:
    #     return None
    # mask = mask & (advantages >= advantages[mask].median())
    if mask.float().mean().item() < 0.1:
        return None

    adv_masked = advantages[mask] #/ nu
    softmax = adv_masked.softmax(0)
    # exp_adv = adv_masked.sub(adv_masked.max()).exp_()
    # softmax = exp_adv / exp_adv.sum()

    # adv_norm = advantages.sign() * advantages.abs()#.pow(nu)

    #kl_mask = (kl <= kl_limit).float()
    loss_policy = softmax.sub(softmax.median()).mul_(softmax.numel()).mul_(vtrace_p[mask]).clamp_(-5, 5).detach_().mul_(-logp[mask]) # * kl_mask
    #loss_nu = nu * eps_nu + nu * adv_masked.exp().mean().log()
    loss_alpha = alpha.detach() * kl #* (kl_mask * 0.9 + 0.1)
    # loss_alpha = alpha * (eps_alpha - kl.detach()) + alpha.detach() * kl #* (kl_mask * 0.9 + 0.1)

    assert loss_policy.shape == adv_masked.shape, (loss_policy.shape, adv_masked.shape)
    #assert loss_nu.shape == (), loss_nu.shape
    assert loss_alpha.shape == kl.shape, (loss_alpha.shape, kl.shape)

    return loss_policy.mean(), torch.scalar_tensor(0.0, device=kl.device), loss_alpha.mean()


@torch.jit.script
def scaled_impala_loss(kl_target: torch.Tensor, kl_replay: torch.Tensor, logp: torch.Tensor, advantages: torch.Tensor, advantages_upgo: torch.Tensor,
                       vtrace_p: torch.Tensor, nu: torch.Tensor, alpha: torch.Tensor,
                       eps_nu: float, eps_alpha: float, kl_limit: float) \
        -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    assert kl_target.dim() == logp.dim() == advantages.dim() == 1
    assert nu.shape == alpha.shape == ()
    # assert eps_nu > 1, eps_nu

    advantages = advantages.mul(vtrace_p).add_(advantages_upgo)

    # old_rms = advantages.pow(2).mean().sqrt_().add_(1e-5)
    # advantages = advantages.abs().pow_(nu).mul_(advantages.sign())
    # advantages *= old_rms / advantages.pow(2).mean().sqrt_().add_(1e-5)

    kl_mask = (kl_target <= kl_limit).float()#.mul_(advantages.abs().clamp_max_(2))
    loss_policy = advantages.clamp(-5, 5).detach_().mul_(-logp).mul_(kl_mask)
    # loss_alpha = alpha * (eps_alpha - kl_target.detach()) + alpha.detach() * kl_target
    loss_alpha = alpha.detach() * kl_target #* kl_mask.clamp_min(0.2)
    # loss_alpha = alpha * (eps_alpha - kl_target.detach()) + alpha.detach() * kl_target

    # mask = vtrace_p > 0.5
    zero = torch.scalar_tensor(0.0, device=nu.device)
    # if mask.float().mean().item() > 0.2:
    #     adv_mask = advantages[mask]
    #     std, mean = torch.std_mean(adv_mask)
    #     kurtosis = adv_mask.sub(mean).div_(std + 1e-5).pow_(4).mean().pow_(1 / 4).detach_()
    #     loss_nu = nu * (kurtosis - eps_nu).sign_()
    # else:
    loss_nu = kurtosis = zero

    assert loss_policy.shape == advantages.shape, (loss_policy.shape, advantages.shape)
    assert loss_nu.shape == (), loss_nu.shape
    assert loss_alpha.shape == kl_target.shape, (loss_alpha.shape, kl_target.shape)

    return loss_policy.mean(), loss_nu.mean(), loss_alpha.mean(), kurtosis


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