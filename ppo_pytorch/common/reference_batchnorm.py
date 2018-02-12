from functools import partial

import torch
import torch.nn as nn
from torch.autograd import Variable
import random


def update_ref_batch(module, input, ref_size=256, ref_replaced_samples=1):
    if isinstance(input, Variable):
        input = input.data

    # ref batch update
    if not hasattr(module, '__ref_batch'):
        module.register_buffer('__ref_batch', None)
    if module.__ref_batch is None:
        module.__ref_batch = input[:min(ref_size, input.shape[0])]

    if module.__ref_batch.shape[0] < ref_size:
        added_samples = min(input.shape[0], ref_size - module.__ref_batch.shape[0])
        module.__ref_batch = torch.cat([module.__ref_batch, input[:added_samples]], 0)
    else:
        if random.random() > 0.1:
            return
        idx_inp = torch.LongTensor(ref_replaced_samples).random_(0, input.shape[0])
        idx_ref = torch.LongTensor(ref_replaced_samples).random_(0, ref_size)
        if input.is_cuda:
            idx_inp, idx_ref = idx_inp.cuda(), idx_ref.cuda()
        module.__ref_batch[idx_ref] = input[idx_inp]


def forward_with_ref_batch(module, input):
    # def enable_ref_update(m, on):
    #     if m.__class__.__name__.find('RefBatchNorm') != -1:
    #         m.train(on)
    #
    # module.apply(lambda m: enable_ref_update(m, True))
    ref = Variable(module.__ref_batch)
    data = torch.cat([input, ref], 0) if input is not None else ref
    out = module(data)
    # module.apply(lambda m: enable_ref_update(m, False))
    if input is not None:
        from ..models.actors import ActorOutput
        if isinstance(out, ActorOutput):
            out = ActorOutput(probs=out.probs[:input.shape[0]], state_values=out.state_values[:input.shape[0]])
        return out


class _RefBatchNorm(nn.Module):
    def __init__(self, num_features, ref_batch_size=256, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.ref_batch_size = ref_batch_size
        self.eps = eps
        self.weight = nn.Parameter(torch.Tensor(num_features))
        self.bias = nn.Parameter(torch.Tensor(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.weight.data.normal_(1, 0.02)
        self.bias.data.zero_()

    def forward(self, input):
        view_size = -1, *((input.dim() - 2) * [1])
        if self.training:
            x = input[-self.ref_batch_size:]
            x = x.transpose(0, 1).contiguous().view(input.shape[1], -1)
            mean = x.mean(-1)
            var = x.var(-1)
            self.running_mean.copy_(mean.data)
            self.running_var.copy_(var.data)
            mean, var = mean.view(view_size), var.view(view_size)
        else:
            mean = Variable(self.running_mean.view(view_size))
            var = Variable(self.running_var.view(view_size))
        weight = self.weight.view(view_size)
        bias = self.bias.view(view_size)
        return (input - mean) / (var + self.eps).sqrt() * weight + bias


class RefBatchNorm1d(_RefBatchNorm):
    pass


class RefBatchNorm2d(_RefBatchNorm):
    pass
