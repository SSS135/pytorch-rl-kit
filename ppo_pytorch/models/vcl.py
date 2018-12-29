import torch
from torch import nn as nn


class VCL(nn.Module):
    def __init__(self,num_features, beta_init=1.0, sample_size=5):
        super().__init__()
        self.num_features = num_features
        self.beta_init = beta_init
        self.sample_size = sample_size
        self.vcl_beta = nn.Parameter(torch.Tensor(num_features))
        self.extra_loss = None
        self.reset_parameters()

    def reset_parameters(self):
        self.vcl_beta.data.fill_(1)

    def forward(self, input):
        self.vcl_beta.data.clamp_(min=1e-3)
        slices = torch.split(input, self.sample_size, dim=0)
        var_a = slices[0].var(dim=0).abs()
        var_b = slices[-1].var(dim=0).abs()
        self.extra_loss = (1 - var_a / (var_b + self.vcl_beta)).pow(2).mean()
        return input

    def extra_repr(self):
        return f'num_features={self.num_features}, beta_init={self.beta_init}, sample_size={self.sample_size}'