import torch
import torch.jit


def blend_models(src, dst, factor):
    for src, dst in zip(src.state_dict().values(), dst.state_dict().values()):
        if dst.dtype == torch.long:
            dst.data.copy_(src.data)
        else:
            dst.data.lerp_(src.data, factor)


@torch.jit.script
def v_mpo_loss(kl: torch.Tensor, logp: torch.Tensor, advantages: torch.Tensor,
               nu: torch.Tensor, alpha: torch.Tensor, eps_nu: float, eps_alpha: float):
    assert kl.dim() == logp.dim() == advantages.dim() == 1
    assert nu.shape == alpha.shape == ()

    mask = advantages >= advantages.median()
    advantages = advantages[mask]
    logp = logp[mask]

    advantages = advantages.clamp(-10, 10) / nu
    max_adv = advantages.detach().max()
    exp_adv = advantages.detach().sub(max_adv).exp()
    softmax = exp_adv / exp_adv.sum()
    loss_policy = softmax * -logp
    loss_nu = nu * eps_nu + nu * advantages.exp().mean().log()
    loss_alpha = alpha * (eps_alpha - kl.detach()) + alpha.detach() * kl

    assert loss_policy.shape == advantages.shape, (loss_policy.shape, advantages.shape)
    assert loss_nu.shape == (), loss_nu.shape
    assert loss_alpha.shape == kl.shape, (loss_alpha.shape, kl.shape)

    return loss_policy.sum(), loss_nu.mean(), loss_alpha.mean()