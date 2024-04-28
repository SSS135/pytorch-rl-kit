import torch
from torch.nn import functional as F


@torch.compile
def symlog(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))


@torch.compile
def symexp(x):
    return torch.sign(x) * torch.expm1(torch.abs(x))


class TwoHotDist:
    def __init__(self, bins: torch.Tensor, transfwd=None, transbwd=None):
        assert bins.dtype.is_floating_point, bins.dtype
        self.bins = torch.asarray(bins)
        self.transfwd = transfwd or (lambda x: x)
        self.transbwd = transbwd or (lambda x: x)

    def mean(self, logits: torch.Tensor):
        wavg = _twohot_mean(logits, self.bins.to(logits.device))
        return self.transbwd(wavg)

    def mode(self, logits: torch.Tensor):
        assert logits.shape[-1] == len(self.bins), (logits.shape, len(self.bins))
        assert logits.dtype.is_floating_point, logits.dtype
        probs = torch.softmax(logits, -1)
        return self.transbwd((probs * self.bins.to(logits.device)).sum(-1))

    def logp(self, x: torch.Tensor, logits: torch.Tensor):
        assert x.dtype.is_floating_point, x.dtype
        x = self.transfwd(x)
        return _twohot_logp(x, logits, self.bins.to(logits.device))


@torch.compile(fullgraph=True, mode='max-autotune')
def _twohot_mean(logits: torch.Tensor, bins: torch.Tensor):
    # The naive implementation results in a non-zero result even if the bins
    # are symmetric and the probabilities uniform, because the sum operation
    # goes left to right, accumulating numerical errors. Instead, we use a
    # symmetric sum to ensure that the predicted rewards and values are
    # actually zero at initialization.
    # return self.transbwd((self.probs * self.bins).sum(-1))

    assert logits.shape[-1] == len(bins), (logits.shape, len(bins))
    assert logits.dtype.is_floating_point, logits.dtype
    probs = torch.softmax(logits, -1)

    n = logits.shape[-1]
    if n % 2 == 1:
        m = (n - 1) // 2
        p1 = probs[..., :m]
        p2 = probs[..., m: m + 1]
        p3 = probs[..., m + 1:]
        b1 = bins[..., :m]
        b2 = bins[..., m: m + 1]
        b3 = bins[..., m + 1:]
        wavg = (p2 * b2).sum(-1) + ((p1 * b1).flip(-1) + (p3 * b3)).sum(-1)
        return wavg
    else:
        p1 = probs[..., :n // 2]
        p2 = probs[..., n // 2:]
        b1 = bins[..., :n // 2]
        b2 = bins[..., n // 2:]
        wavg = ((p1 * b1).flip(-1) + (p2 * b2)).sum(-1)
        return wavg


@torch.compile(fullgraph=True, mode='max-autotune')
def _twohot_logp(x: torch.Tensor, logits: torch.Tensor, bins: torch.Tensor):
    # below = (bins <= x[..., None]).sum(-1) - 1
    # above = len(bins) - (bins > x[..., None]).sum(-1)
    # below = torch.clip(below, 0, len(bins) - 1)
    # above = torch.clip(above, 0, len(bins) - 1)
    # equal = (below == above)
    # dist_to_below = _where(equal, 1, torch.abs(bins[below] - x))
    # dist_to_above = _where(equal, 1, torch.abs(bins[above] - x))
    # total = dist_to_below + dist_to_above
    # weight_below = dist_to_above / total
    # weight_above = dist_to_below / total
    # target = (F.one_hot(below, len(bins)) * weight_below[..., None] +
    #           F.one_hot(above, len(bins)) * weight_above[..., None])
    # log_pred = logits - torch.logsumexp(logits, -1, keepdim=True)
    # return (target * log_pred).sum(-1)

    assert bins.shape == (logits.shape[-1],), (bins.shape, logits.shape)
    assert x.shape == logits.shape[:-1], (x.shape, logits.shape)

    x = x[..., None]
    # below in [-1, len(bins) - 1]
    below = (bins <= x).type(torch.int32).sum(dim=-1, keepdim=True) - 1
    # above in [0, len(bins)]
    above = below + 1

    # above in [0, len(bins) - 1]
    above = torch.minimum(above, torch.full_like(above, len(bins) - 1))
    # below in [0, len(bins) - 1]
    below = torch.maximum(below, torch.zeros_like(below))

    equal = below == above
    dist_to_below = torch.where(equal, 1, torch.abs(bins[below] - x))
    dist_to_above = torch.where(equal, 1, torch.abs(bins[above] - x))
    total = dist_to_below + dist_to_above
    weight_below = dist_to_above / total
    weight_above = dist_to_below / total
    target = (
            F.one_hot(below, len(bins)) * weight_below[..., None]
            + F.one_hot(above, len(bins)) * weight_above[..., None]
    ).squeeze(-2)
    log_pred = logits - torch.logsumexp(logits, dim=-1, keepdims=True)
    assert target.shape == log_pred.shape, (target.shape, log_pred.shape)
    return (target * log_pred).sum(-1)


@torch.compile(fullgraph=True)
def _where(cond, true, false):
    return cond * true + (1 - cond) * false