import torch
from torch import jit


@jit.script
def squash(x, eps: float = 0.001):
    return x.sign() * ((x.abs() + 1).sqrt() - 1) + eps * x


@jit.script
def unsquash(x, eps: float = 0.001):
    return x.sign() * ((((1 + 4 * eps * (x.abs() + 1 + eps)).sqrt() - 1) / (2 * eps)) ** 2 - 1)


def test_squash():
    torch.manual_seed(123)
    t = torch.randn(1000) * 100
    assert torch.allclose(t, squash(unsquash(t)), atol=1e-3), (t, squash(unsquash(t)))
    assert torch.allclose(t, unsquash(squash(t)), atol=1e-3), (t, unsquash(squash(t)))