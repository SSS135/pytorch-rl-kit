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
    advmax = 20 * nu.item()
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