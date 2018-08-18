import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class OptClipFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, pivot, clip, grad_old=None):
        assert x.shape == pivot.shape and clip.dim() == 0
        ctx.save_for_backward(x, pivot, clip)
        ctx.grad_old = grad_old
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        x, pivot, clip = ctx.saved_tensors
        grad_old = ctx.grad_old
        grad_output = grad_output.clone()
        if grad_old is not None:
            clip = grad_old.abs() * clip
        diff = (x - pivot) / clip
        zero_mask = (diff > 1) & (grad_output < 0) | (diff < -1) & (grad_output > 0)
        grad_output[zero_mask] = 0
        return grad_output, None, None, None


def opt_clip(x, pivot, clip, grad_old=None):
    if not isinstance(clip, torch.Tensor):
        clip = torch.tensor(clip, dtype=x.dtype, device=x.device)
    return OptClipFunction.apply(x, pivot, clip, grad_old)


def clipped_loss(x, x_old, clip, loss_calc):
    x_old = x_old.detach()
    x_old.requires_grad = True
    x_old_loss = loss_calc(x_old)
    x_old_grad = torch.autograd.grad(x_old_loss.sum(), x_old)[0]
    x_clip = opt_clip(x, x_old, clip, x_old_grad)
    loss_x = loss_calc(x_clip)
    return loss_x
