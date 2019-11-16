import torch
import torch.jit


def blend_models(src, dst, factor):
    for src, dst in zip(src.state_dict().values(), dst.state_dict().values()):
        if dst.dtype == torch.long:
            dst.data.copy_(src.data)
        else:
            dst.data.lerp_(src.data, factor)


@torch.jit.script
def v_mpo_loss(kl, logp, advantages, nu, alpha, eps_nu: float, eps_alpha: float):
    adv_clamp = advantages.clamp(-10, 10)
    top_mask = adv_clamp >= adv_clamp.median()
    top_advantages = adv_clamp[top_mask]
    exp_top_advantages = top_advantages.div(nu).exp()
    max_adv = adv_clamp.max()
    softmax = adv_clamp.sub(max_adv).div(nu).exp().unsqueeze(-1) / \
              top_advantages.sub(max_adv).div(nu).exp().sum()
    loss_policy = (softmax.detach() * -logp).mean(-1) * top_mask.float()
    loss_nu = nu * eps_nu + nu * exp_top_advantages.mean().log()
    loss_alpha = alpha * (eps_alpha - kl.detach()) + alpha.detach() * kl

    assert loss_policy.shape == kl.shape[:-1], (loss_policy.shape, kl.shape)
    assert loss_nu.shape == (), loss_nu.shape
    assert loss_alpha.shape[:-1] == loss_policy.shape and loss_alpha.shape[-1] == kl.shape[-1], (loss_alpha.shape, loss_policy.shape)

    return loss_policy.sum(), loss_nu.mean(), loss_alpha.mean()