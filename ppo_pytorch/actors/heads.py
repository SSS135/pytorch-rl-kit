import math
import random

import torch.nn as nn
import torch
import torch.nn.functional as F

from .utils import normalized_columns_initializer_
from ..common.probability_distributions import ProbabilityDistribution, CategoricalPd, MultiCategoricalPd, \
    BetaPd, FixedStdGaussianPd, DiagGaussianPd, LinearTanhPd, DiscretizedCategoricalPd
from ..common.attr_dict import AttrDict
from ..config import Linear


def assert_shape(x, shape, *args):
    assert x.shape == shape, (tuple(x.shape), shape, *args)


class HeadBase(nn.Module):
    """
    Base class for output of `Actor`. Different heads used for different learning algorithms.
    """

    def __init__(self, in_features):
        """
        Args:
            in_features: Input feature vector width.
        """
        super().__init__()
        self.in_features = in_features

    def reset_weights(self):
        raise NotImplementedError()


class ActionValuesHead(HeadBase):
    """
    Head with state-values. Used in Q learning. Support dueling architecture.
    """

    def __init__(self, in_features, pd: CategoricalPd, dueling=False):
        """
        Args:
            in_features: Input feature vector width.
            pd: Action probability distribution.
            dueling: Use dueling architecture.
        """
        super().__init__(in_features)
        assert isinstance(pd, CategoricalPd)
        self.pd = pd
        self.dueling = dueling
        self.linear = Linear(in_features, self.pd.prob_vector_len + 1)
        self.reset_weights()

    def reset_weights(self):
        normalized_columns_initializer_(self.linear.weight.data, 1.0)
        self.linear.bias.data.fill_(0)

    def forward(self, x, **kwargs):
        A, V = self.linear(x).split([self.pd.prob_vector_len, 1], -1)
        if self.dueling:
            Q = V + (A - A.mean(-1, keepdim=True))
        else:
            Q = A
        Q = Q.contiguous()
        return Q


class PolicyHead(HeadBase):
    """
    Actor-critic head. Used in PPO / A3C.
    """

    def __init__(self, in_features, pd: ProbabilityDistribution):
        """
        Args:
            in_features: Input feature vector width.
            pd: Action probability distribution.
        """
        super().__init__(in_features)
        self.pd = pd
        self.linear = Linear(in_features, self.pd.prob_vector_len)
        self.reset_weights()

    def reset_weights(self):
        normalized_columns_initializer_(self.linear.weight.data, self.pd.init_column_norm)
        self.linear.bias.data.fill_(0)

    def forward(self, x, **kwargs):
        return self.linear(x)


class PositionalPolicyHead(HeadBase):
    """
    Actor-critic head. Used in PPO / A3C.
    """

    def __init__(self, in_features, pd: DiscretizedCategoricalPd, num_reduced=32):
        """
        Args:
            in_features: Input feature vector width.
            pd: Action probability distribution.
        """
        super().__init__(in_features)
        self.pd = pd
        assert pd.prob_vector_len % pd.d == 0
        self.linear_final = Linear(num_reduced, pd.d)
        self.linear_reduce = Linear(in_features, num_reduced)
        self.pos_embedding = Linear(1, num_reduced)
        self.reset_weights()

    def reset_weights(self):
        normalized_columns_initializer_(self.linear_final.weight.data, self.pd.init_column_norm)
        self.linear_final.bias.data.fill_(0)
        self.pos_embedding.reset_parameters()
        self.linear_reduce.reset_parameters()

    def forward(self, x, **kwargs):
        pos = torch.linspace(-1, 1, self.pd.num_bins, device=x.device).view(-1, 1)
        pos = self.pos_embedding(pos).sigmoid()
        x = F.relu(self.linear_reduce(x))
        x = x.unsqueeze(-2).expand(*x.shape[:-1], self.pd.num_bins, x.shape[-1])
        # (*, num_bins, features)
        x = pos * x
        # (*, num_bins, d)
        logits = self.linear_final(x)
        # (*, d * num_bins)
        logits = logits.transpose(-1, -2).flatten(start_dim=-2)
        return logits


class RepeatPolicyHead(HeadBase):
    """
    Actor-critic head. Used in PPO / A3C.
    """

    def __init__(self, in_features, num_repeats, pd: ProbabilityDistribution):
        """
        Args:
            in_features: Input feature vector width.
            pd: Action probability distribution.
        """
        super().__init__(in_features)
        self.pd = pd
        self.num_repeats = num_repeats
        self.linear = Linear(in_features, self.pd.prob_vector_len // num_repeats)
        self.reset_weights()

    def reset_weights(self):
        normalized_columns_initializer_(self.linear.weight.data, self.pd.init_column_norm)
        self.linear.bias.data.fill_(0)

    def forward(self, x, **kwargs):
        assert x.shape[-2:] == (self.num_repeats, self.in_features)
        x = self.linear(x)
        if isinstance(self.pd, BetaPd) or isinstance(self.pd, DiagGaussianPd):
            x = x.transpose(-1, -2)
        x = x.reshape(*x.shape[:-2], self.pd.prob_vector_len)
        return x


class StateValueHead(HeadBase):
    """
    Actor-critic head. Used in PPO / A3C.
    """

    def __init__(self, in_features, pd: ProbabilityDistribution = None, num_out=1):
        """
        Args:
            in_features: Input feature vector width.
            pd: Action probability distribution.
        """
        super().__init__(in_features)
        self.pd = pd
        self.num_out = num_out
        self.linear = Linear(in_features, num_out)
        self.reset_weights()

    def reset_weights(self):
        normalized_columns_initializer_(self.linear.weight.data, 1.0)
        self.linear.bias.data.fill_(0)

    def forward(self, x, **kwargs):
        # (*xd, bins, 1)
        return self.linear(x)

    def normalize(self, mean, std):
        self.linear.bias.data -= mean
        self.linear.bias.data /= std
        self.linear.weight.data /= std

    def unnormalize(self, mean, std):
        self.linear.weight.data *= std
        self.linear.bias.data *= std
        self.linear.bias.data += mean