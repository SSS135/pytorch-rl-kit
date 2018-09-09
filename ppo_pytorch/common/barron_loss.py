import math

import torch


def barron_loss(pred, target, alpha, c, reduce=True):
    """
    A More General Robust Loss Function
    https://arxiv.org/abs/1701.03077
    """
    assert isinstance(alpha, float) or isinstance(alpha, int)
    assert isinstance(c, float) or isinstance(c, int)
    assert pred.shape == target.shape

    mse = (target - pred).div_(c).pow_(2)
    if alpha == 2:
        loss = 0.5 * mse
    elif alpha == 0:
        loss = torch.log(0.5 * mse + 1)
    elif alpha == -math.inf:
        loss = 1 - torch.exp(-0.5 * mse)
    else:
        scale = abs(2 - alpha) / alpha
        inner = mse / abs(2 - alpha) + 1
        loss = scale * (inner ** (alpha / 2) - 1)
    return loss.mean() if reduce else loss


def barron_loss_derivative(x, alpha, c):
    """
    A More General Robust Loss Function
    https://arxiv.org/abs/1701.03077
    """
    assert isinstance(alpha, float) or isinstance(alpha, int)
    assert isinstance(c, float) or isinstance(c, int)

    if alpha == 2:
        return x / c ** 2
    elif alpha == 0:
        return 2 * x / (x ** 2 + 2 * c ** 2)
    elif alpha == -math.inf:
        return x / c ** 2 * torch.exp(-0.5 * (x / c) ** 2)
    else:
        scale = x / c ** 2
        inner = (x / c) ** 2 / abs(2 - alpha) + 1
        return scale * inner ** (alpha / 2 - 1)


def pseudo_huber_loss(pred, target, reduce=True):
    assert pred.shape == target.shape
    loss = ((target - pred) ** 2 + 1).sqrt() - 1
    return loss.mean() if reduce else loss