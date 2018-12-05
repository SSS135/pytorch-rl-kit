import math
import torch
from ppo_pytorch.common.probability_distributions import CategoricalPd, DiagGaussianPd, BetaPd
from torch.nn.utils import clip_grad_norm_
import numpy as np
import torch.optim
from .barron_loss import barron_loss, barron_loss_derivative


def get_target_logits(pd, action, logits, change, nsteps=20, lr=0.4, lr_pow=0.8, kl_scale=1.0, rms_alpha=0.8):
    assert logits.dim() - 1 == change.dim() >= 1
    target = change.detach().unsqueeze(-1)
    old_logits = logits.detach()
    logits = old_logits.clone()
    logits.requires_grad = True
    old_logp = pd.logp(action, logits.detach())

    optim = torch.optim.RMSprop([logits], lr=0.0, alpha=rms_alpha, eps=0.1)

    for i in range(nsteps):
        with torch.enable_grad():
            new_logp = pd.logp(action, logits)
            loss = barron_loss((new_logp - old_logp), target.expand_as(old_logp), reduce=False).sum() + kl_scale * pd.kl(old_logits, logits).pow(2).sum()

        loss.backward()

        max_norm = 5
        logits.grad.clamp_(-1e3, 1e3)
        logits.grad /= logits.grad.abs().max(-1, keepdim=True)[0].clamp_(min=max_norm).div_(max_norm)
        optim.param_groups[0]['lr'] = lr * lr_pow ** i
        optim.step()

        # diff = logits.data - old_logits.data
        # diff_rms = diff.pow(2).mean(-1, keepdim=True).sqrt_().clamp_(min=1e-3)
        # rev = 1 if i + 1 == nsteps else (1 - lr * lr_pow ** (i + 1))
        # logits.data = old_logits.data + rev * target.abs() / diff_rms * diff

        # print('step', i, 'lr', lr * lr_pow ** i)
        # print('data', logits.data)
        # print('grad', logits.grad)
        # print('logit diff', (logits.data - old_logits).squeeze())
        # print('prob diff', ((pd.logp(action, logits) - old_logp) - target).squeeze())

        optim.zero_grad()

    return logits.detach()


def test_logits():
    with torch.no_grad():
        pd_factories = [lambda x: DiagGaussianPd(x // 2), lambda x: BetaPd(x // 2, 1), lambda x: CategoricalPd(x)]
        targets = torch.tensor([-0.3, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.3]).cuda()
        pd_sizes = [2, 4, 8, 16, 32]
        batch_size = 10 * 1024

        quality_list = []

        print()

        pds = [pd_f(size) for pd_f in pd_factories for size in pd_sizes]
        for pd in pds:
            src_logits = barron_loss_derivative(torch.randn(batch_size, len(targets), pd.prob_vector_len).cuda())
            cur_targets = targets.view(1, -1).repeat(batch_size, 1)
            cur_targets *= barron_loss_derivative(torch.empty_like(cur_targets).normal_())
            actions = pd.sample(src_logits)
            out_logits = get_target_logits(pd, actions, src_logits, cur_targets)
            src_logp = pd.logp(actions, src_logits)
            out_logp = pd.logp(actions, out_logits)
            diff_logp = out_logp - src_logp

            out_rms_error = (diff_logp - cur_targets.unsqueeze(-1)).view(-1).abs()
            src_rms_error = cur_targets.unsqueeze(-1).add(0 * diff_logp).view(-1).abs()
            quality = 1 - (out_rms_error / src_rms_error.clamp_(min=1e-2))
            sign_check = ((diff_logp > 0) == (cur_targets.unsqueeze(-1) > 0)).float().mean().item()
            kl = pd.kl(src_logits, out_logits)
            max_logit_diff = (src_logits - out_logits).abs().max().item()

            quality_list.append(quality.mean().item())

            print(
                pd,
                ', quality', round(quality.mean().item(), 4),
                ', abs error', round(out_rms_error.mean().item(), 4),
                ', max error', round(out_rms_error.max().item(), 4),
                ', dir check', round(sign_check, 4),
                ', avg kl', round(kl.mean().item(), 4),
                ', max kl', round(kl.max().item(), 4),
                ', max logit diff', round(max_logit_diff, 4),
            )

        quality_list = np.array(quality_list)
        print('quality mean', quality_list.mean().round(4),
              'median', np.median(quality_list).round(4),
              'min', np.min(quality_list).round(4))
