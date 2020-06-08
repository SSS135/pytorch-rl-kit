from torch import nn


def silu(x):
    return x * x.sigmoid()


class SiLU(nn.Module):
    def forward(self, x):
        return silu(x)