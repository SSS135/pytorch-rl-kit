from typing import Optional, Tuple

import torch
import torch.jit


def blend_models(src, dst, factor):
    for src, dst in zip(src.state_dict().values(), dst.state_dict().values()):
        if dst.dtype == torch.long:
            dst.data.copy_(src.data)
        else:
            dst.data.lerp_(src.data, factor)


@torch.jit.script
def v_mpo_loss(kl: torch.Tensor, logp: torch.Tensor, advantages: torch.Tensor, advantages_upgo: torch.Tensor, vtrace_p: torch.Tensor,
               nu: torch.Tensor, alpha: torch.Tensor, eps_nu: float, eps_alpha: float) \
        -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    assert kl.dim() == logp.dim() == advantages.dim() == 1
    assert nu.shape == alpha.shape == ()

    advantages = advantages.mul(vtrace_p).masked_fill(vtrace_p < 0.01, float('-inf'))
    advmax = 5 * nu.item()
    advantages = advantages.clamp(-advmax, advmax) / nu
    advantages_upgo = advantages_upgo.clamp(-advmax, advmax) / nu

    test_adv = advantages.detach().exp() + advantages_upgo.detach().exp()
    mask = test_adv >= test_adv.median()
    advantages = advantages[mask]
    advantages_upgo = advantages_upgo[mask]
    logp = logp[mask]

    max_adv = torch.max(advantages.detach().max(), advantages_upgo.detach().max())
    exp_adv = advantages.detach().sub(max_adv).exp() + advantages_upgo.detach().sub(max_adv).exp()
    softmax = exp_adv / exp_adv.sum()

    loss_policy = softmax * -logp
    loss_nu = nu * eps_nu + nu * (advantages.exp() + advantages_upgo.exp()).mean().log()
    loss_alpha = alpha * (eps_alpha - kl.detach()) + alpha.detach() * kl

    assert loss_policy.shape == advantages.shape, (loss_policy.shape, advantages.shape)
    assert loss_nu.shape == (), loss_nu.shape
    assert loss_alpha.shape == kl.shape, (loss_alpha.shape, kl.shape)

    return loss_policy.sum(), loss_nu.mean(), loss_alpha.mean()


@torch.jit.script
def scaled_impala_loss(kl: torch.Tensor, logp: torch.Tensor, advantages: torch.Tensor, advantages_upgo: torch.Tensor, vtrace_p: torch.Tensor,
               nu: torch.Tensor, alpha: torch.Tensor, eps_nu: float, eps_alpha: float) \
        -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    assert kl.dim() == logp.dim() == advantages.dim() == 1
    assert nu.shape == alpha.shape == ()

    advantages = advantages.mul(vtrace_p)
    advantages = (advantages + advantages_upgo).clamp(-10, 10)

    # adv_mean = advantages.mean()
    # adv_std = advantages.std() + 1e-5
    # adv_norm = (advantages - adv_mean) / adv_std
    adv_norm = advantages.sign() * advantages.abs().pow(nu)

    # adv_fix = adv_norm # adv_norm * adv_std + adv_mean
    loss_policy = adv_norm.detach() * -logp
    loss_nu = nu * (adv_norm.detach().pow(4).mean() - eps_nu)
    loss_alpha = alpha * (eps_alpha - kl.detach()) + alpha.detach() * kl

    assert loss_policy.shape == advantages.shape, (loss_policy.shape, advantages.shape)
    assert loss_nu.shape == (), loss_nu.shape
    assert loss_alpha.shape == kl.shape, (loss_alpha.shape, kl.shape)

    return loss_policy.sum(), loss_nu.mean(), loss_alpha.mean()


class RunningNorm:
    def __init__(self, momentum=0.99, mean_norm=True):
        self.momentum = momentum
        self.mean_norm = mean_norm
        self._stats = (0, 0, 0)

    def __call__(self, values, update_stats=True):
        mean, square, iter = self._stats
        if update_stats:
            mean = self.momentum * mean + (1 - self.momentum) * values.mean().item()
            square = self.momentum * square + (1 - self.momentum) * values.pow(2).mean().item()
            iter += 1
            self._stats = (mean, square, iter)

        bias_corr = 1 - self.momentum ** iter
        mean = mean / bias_corr
        square = square / bias_corr

        if self.mean_norm:
            std = (square - mean ** 2) ** 0.5
            values = (values - mean) / max(std, 1e-5)
        else:
            rms = square ** 0.5
            values = values / max(rms, 1e-5)

        return values