import random
from typing import Optional

import torch
import torch.nn as nn


class ActivationNorm(nn.Module):
    def __init__(self, data_dims, eps=1e-5):
        super().__init__()
        self.data_dims = data_dims
        self.eps = eps
        self._input = None

    def forward(self, input: torch.Tensor, always_run=False):
        if not always_run and not input.requires_grad:
            return input
        assert input.dim() > self.data_dims
        self._input = input
        return input.clone()

    def get_loss(self, clear=True) -> Optional[torch.Tensor]:
        if self._input is None:
            return None
        var, mean = torch.var_mean(self._input, dim=tuple(range(self._input.ndim - self.data_dims)))
        var = var + self.eps
        loss = 0.5 * (var.mean() + mean.pow(2).mean() - var.log().mean())
        if clear:
            self._input = None
        return loss


# class ActivationNormFunction(torch.autograd.Function):
#     """
#     We can implement our own custom autograd Functions by subclassing
#     torch.autograd.Function and implementing the forward and backward passes
#     which operate on Tensors.
#     """
#
#     @staticmethod
#     def forward(ctx, input):
#         """
#         In the forward pass we receive a Tensor containing the input and return
#         a Tensor containing the output. ctx is a context object that can be used
#         to stash information for backward computation. You can cache arbitrary
#         objects for use in the backward pass using the ctx.save_for_backward method.
#         """
#         ctx.mark_dirty(input)
#         ctx.save_for_backward(input)
#         return input
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         """
#         In the backward pass we receive a Tensor containing the gradient of the loss
#         with respect to the output, and we need to compute the gradient of the loss
#         with respect to the input.
#         """
#         input, = ctx.saved_tensors
#         grad_input = grad_output.clone()
#         grad_input[input < 0] = 0
#         return grad_input


class ActivationNormWrapper(nn.Module):
    def __init__(self, data_dims, wrapped_module):
        super().__init__()
        self.wrapped_module = wrapped_module
        self.act_norm = ActivationNorm(data_dims)

    def forward(self, input: torch.Tensor):
        output = self.wrapped_module(input)
        if input.requires_grad:
            self.act_norm(self.wrapped_module(input.detach()), always_run=True)
        return output


def activation_norm_loss(module):
    losses = []
    for m in module.modules():
        if isinstance(m, ActivationNorm):
            loss = m.get_loss()
            if loss is not None:
                losses.append(loss.view(1))
    return torch.cat(losses).mean() if len(losses) > 0 else torch.scalar_tensor(0.0)
