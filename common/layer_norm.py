import torch
import torch.nn as nn


def layer_norm_1d(input, bias=None, scale=None, eps=1e-4):
    assert (bias is None) == (scale is None)
    mean = input.mean(-1, keepdim=True)
    std = input.std(-1, keepdim=True).clamp(min=eps)
    input = (input - mean) / std
    if bias is not None and scale is not None:
        input = torch.addcmul(bias, tensor1=input, tensor2=scale)
    return input


def layer_norm_2d(input, bias=None, scale=None, eps=1e-4):
    assert (bias is None) == (scale is None)

    # x = input.view(input.size(0) * input.size(1), -1)
    x = input.view(input.shape[0], -1)
    mean = x.mean(-1, keepdim=True)
    std = x.std(-1, keepdim=True).clamp(min=eps)
    x = (x - mean) / std
    input = x.view_as(input)

    if bias is not None and scale is not None:
        input = torch.addcmul(bias.view(-1, 1, 1), tensor1=input, tensor2=scale.view(-1, 1, 1))
    return input


class LayerNorm1d(nn.Module):
    def __init__(self, features=None, eps=1e-4):
        super().__init__()
        self.features = features
        self.eps = eps
        if features is not None:
            self.weight = nn.Parameter(torch.ones(features))
            self.bias = nn.Parameter(torch.zeros(features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.features is not None:
            self.weight.data.fill_(1)
            self.bias.data.zero_()

    def forward(self, x):
        return layer_norm_1d(x, self.bias, self.weight, self.eps)


class LayerNorm2d(LayerNorm1d):
    def forward(self, x):
        return layer_norm_2d(x, self.bias, self.weight, self.eps)