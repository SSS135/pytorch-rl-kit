from functools import partial

import torch
import torch.nn as nn
from torch.autograd import Variable


def update_ref_batch_norm(module, input=None, update_momentum=True, ref_size=256, ref_replaced_samples=256):
    # if isinstance(input, Variable):
    #     input = input.data
    #
    # # ref batch update
    # if not hasattr(module, '__ref_batch'):
    #     module.register_buffer('__ref_batch', None)
    # if module.__ref_batch is None and input is not None:
    #     module.__ref_batch = input[:min(ref_size, input.shape[0])]
    #
    # if input is None and module.__ref_batch is None:
    #     return
    #
    # if input is not None:
    #     if module.__ref_batch.shape[0] < ref_size:
    #         added_samples = min(input.shape[0], ref_size - module.__ref_batch.shape[0])
    #         module.__ref_batch = torch.cat([module.__ref_batch, input[:added_samples]], 0)
    #     else:
    #         idx_inp = torch.LongTensor(ref_replaced_samples).random_(0, input.shape[0])
    #         idx_ref = torch.LongTensor(ref_replaced_samples).random_(0, ref_size)
    #         if input.is_cuda:
    #             idx_inp, idx_ref = idx_inp.cuda(), idx_ref.cuda()
    #         module.__ref_batch[idx_ref] = input[idx_inp]
    #         # replacement = input[idx]
    #         # idx.random_(0, ref_size)
    #         # module.__ref_batch[idx] = replacement

    # momentum update
    if update_momentum:
        def enable_ref_update(m, on):
            if m.__class__.__name__.find('BatchNorm') != -1:
                if on:
                    m.momentum = 0
                else:
                    m.momentum = 1

        module.apply(lambda m: enable_ref_update(m, True))
        # module(Variable(module.__ref_batch))
        module(Variable(input))
        module.apply(lambda m: enable_ref_update(m, False))


class RefBatchNorm2d(nn.Module):
    def __init__(self, num_features, momentum=0.9, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.weight = nn.Parameter(torch.Tensor(num_features))
        self.bias = nn.Parameter(torch.Tensor(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.weight.data.uniform_()
        self.bias.data.zero_()

    def forward(self, input):
        # if self.momentum < 1 and self.training:
        if self.training:
            x = input.transpose(0, 1).contiguous().view(input.shape[1], -1)
            mean = x.mean(-1)
            var = x.var(-1)
            self.running_mean.lerp_(mean.data, 1 - self.momentum)
            self.running_var.lerp_(var.data, 1 - self.momentum)
            mean, var = mean.view(-1, 1, 1), var.view(-1, 1, 1)
        else:
            mean = Variable(self.running_mean.view(-1, 1, 1))
            var = Variable(self.running_var.view(-1, 1, 1))
        weight = self.weight.view(-1, 1, 1)
        bias = self.bias.view(-1, 1, 1)
        return (input - mean) / (var + self.eps).sqrt() * weight + bias
