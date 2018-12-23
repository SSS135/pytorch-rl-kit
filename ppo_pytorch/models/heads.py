import math
import random

import torch.nn as nn
import torch
import torch.nn.functional as F

from .utils import normalized_columns_initializer_
from ..common.probability_distributions import ProbabilityDistribution, CategoricalPd
from ..common.attr_dict import AttrDict


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
        self.linear = nn.Linear(in_features, self.pd.prob_vector_len + 1)
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
        self.linear = nn.Linear(in_features, self.pd.prob_vector_len)
        self.reset_weights()

    def reset_weights(self):
        normalized_columns_initializer_(self.linear.weight.data, self.pd.init_column_norm)
        self.linear.bias.data.fill_(0)

    def forward(self, x, **kwargs):
        return self.linear(x)


class StateValueHead(HeadBase):
    """
    Actor-critic head. Used in PPO / A3C.
    """

    def __init__(self, in_features, pd: ProbabilityDistribution = None, num_bins=1):
        """
        Args:
            in_features: Input feature vector width.
            pd: Action probability distribution.
        """
        super().__init__(in_features)
        self.pd = pd
        self.num_bins = num_bins
        self.linear = nn.Linear(in_features, num_bins)
        self.action_linear = nn.Linear(pd.input_vector_len, in_features) if pd is not None else None
        self.reset_weights()

    def reset_weights(self):
        normalized_columns_initializer_(self.linear.weight.data, 1.0)
        self.linear.bias.data.fill_(0)

    def forward(self, x, actions=None, **kwargs):
        if actions is not None:
            x = x * self.action_linear(self.pd.to_inputs(actions)).sigmoid()
        return self.linear(x).unsqueeze(-1)

    def normalize(self, mean, std):
        self.linear.weight.data /= std
        self.linear.bias.data -= mean
        self.linear.bias.data /= std

    def unnormalize(self, mean, std):
        self.linear.weight.data *= std
        self.linear.bias.data *= std
        self.linear.bias.data += mean


class StateValueQuantileHead(StateValueHead):
    def __init__(self, in_features, num_bins=1, tau_dim=32):
        super().__init__(in_features, num_bins)
        self.tau_dim = tau_dim
        self.tau_embedding = nn.Linear(tau_dim * 2, in_features)
        self.reset_weights()

    def forward(self, x, tau=None, **kwargs):
        # x - (*xd, n)
        # tau - (*xd, q * 2)
        xd = x.shape[:-1]
        num_q = tau.shape[-1] // 2
        cur_tau, prev_tau = tau.chunk(2, -1)
        assert xd == cur_tau.shape[:-1], (x.shape, cur_tau.shape)

        # (*xd, q, emb)
        arange = torch.arange(1, 1 + self.tau_dim, device=x.device, dtype=x.dtype).expand(*cur_tau.shape, -1)
        # (*xd, q, emb * 2)
        cur_cos_vec = torch.cos(math.pi * arange * cur_tau.unsqueeze(-1))
        prev_cos_vec = torch.cos(math.pi * arange * prev_tau.unsqueeze(-1))
        assert_shape(cur_cos_vec, (*xd, num_q, self.tau_dim))
        # (*xd, q, emb * 3)
        all_vec = torch.cat([cur_cos_vec, prev_cos_vec], -1)
        assert_shape(all_vec, (*xd, num_q, self.tau_dim * 2))
        # (*xd, q, n)
        tau_emb = F.relu(self.tau_embedding(all_vec))
        assert_shape(tau_emb, (*xd, num_q, x.shape[-1]))

        # (*xd, bins, q)
        res = self.linear(x.unsqueeze(-2) * tau_emb).transpose(-1, -2)
        assert_shape(res, (*xd, self.num_bins, num_q))
        return res