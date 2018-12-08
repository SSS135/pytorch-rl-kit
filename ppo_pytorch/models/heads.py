import math

import torch.nn as nn
import torch
import torch.nn.functional as F

from .utils import normalized_columns_initializer_
from ..common.probability_distributions import ProbabilityDistribution, CategoricalPd
from ..common.attr_dict import AttrDict


class HeadOutput(AttrDict):
    pass


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

    def forward(self, x):
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

    def forward(self, x):
        return self.linear(x)


class StateValueHead(HeadBase):
    """
    Actor-critic head. Used in PPO / A3C.
    """

    def __init__(self, in_features):
        """
        Args:
            in_features: Input feature vector width.
            pd: Action probability distribution.
        """
        super().__init__(in_features)
        self.linear = nn.Linear(in_features, 1)
        self.reset_weights()

    def reset_weights(self):
        normalized_columns_initializer_(self.linear.weight.data, 1.0)
        self.linear.bias.data.fill_(0)

    def forward(self, x):
        return self.linear(x)

    def normalize(self, mean, std):
        self.linear.weight.data /= std
        self.linear.bias.data -= mean
        self.linear.bias.data /= std

    def unnormalize(self, mean, std):
        self.linear.weight.data *= std
        self.linear.bias.data *= std
        self.linear.bias.data += mean


class StateValueQuantileHead(HeadBase):
    """
    Actor-critic head. Used in PPO / A3C.
    """

    def __init__(self, in_features, quantile_dim=32):
        """
        Args:
            in_features: Input feature vector width.
            pd: Action probability distribution.
        """
        super().__init__(in_features)
        self.quantile_dim = quantile_dim
        self.linear = nn.Linear(in_features, 1)
        self.quantile_embedding = nn.Linear(quantile_dim * 3, in_features)
        self.tau = None
        self.reset_weights()

    def reset_weights(self):
        normalized_columns_initializer_(self.linear.weight.data, 1.0)
        self.linear.bias.data.fill_(0)

    def forward(self, x):
        # x - (*xd, n)
        # tau - (*xd, c * 3)
        cur_prev_tau, prev_value = self.tau.split([self.quantile_dim * 2, self.quantile_dim], -1)
        assert x.shape[:-1] == prev_value.shape[:-1], (x.shape, prev_value.shape)

        # (*xd, c, emb)
        arange = torch.arange(1, 1 + self.quantile_dim, device=x.device, dtype=x.dtype).expand(*prev_value.shape, -1)
        # (*xd, c, emb * 2)
        cos_vec = torch.cos(math.pi * arange * cur_prev_tau.unsqueeze(-1))
        # (*xd, c, emb // 2)
        value_vec = prev_value.unsqueeze(-1).expand(*cos_vec.shape[:-1], self.quantile_dim // 2)
        # (*xd, c, emb * 3)
        all_vec = torch.cat([cos_vec, -value_vec, value_vec], -1)
        # (*xd, c, n)
        tau_emb = F.relu(self.quantile_embedding(all_vec))
        # (*xd, c)
        return self.linear(x.unsqueeze(-2) * tau_emb).squeeze(-1)

    def normalize(self, mean, std):
        self.linear.weight.data /= std
        self.linear.bias.data -= mean
        self.linear.bias.data /= std

    def unnormalize(self, mean, std):
        self.linear.weight.data *= std
        self.linear.bias.data *= std
        self.linear.bias.data += mean