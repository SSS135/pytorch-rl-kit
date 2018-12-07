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
        return self.linear(x).squeeze(-1)

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

    def __init__(self, in_features, quantile_dim=64):
        """
        Args:
            in_features: Input feature vector width.
            pd: Action probability distribution.
        """
        super().__init__(in_features)
        self.quantile_dim = quantile_dim
        self.linear = nn.Linear(in_features, 1)
        self.quantile_embedding = nn.Linear(quantile_dim, in_features)
        self.tau = None
        self.reset_weights()

    def reset_weights(self):
        normalized_columns_initializer_(self.linear.weight.data, 1.0)
        self.linear.bias.data.fill_(0)

    def create_tau(self, x, count):
        return torch.rand((count, *x.shape[:-1]), device=x.device, dtype=x.dtype)

    def forward(self, x):
        # x - (*xd, n)
        # tau - (*td, *xd)
        tau = self.tau
        self.tau = None
        assert tau is not None
        assert x.shape[:-1] == tau.shape[-x.dim() + 1:]

        # (*td, *xd, emb)
        arange = torch.arange(self.quantile_dim, device=x.device, dtype=x.dtype).expand(*tau.shape, -1)
        # (*td, *xd, emb)
        cos_vec = torch.cos(math.pi * arange * tau.unsqueeze(-1))
        # (*td, *xd, n)
        tau_emb = F.relu(self.quantile_embedding(cos_vec))
        # (*td, *xd)
        return self.linear(x * tau_emb).squeeze(-1)

    def normalize(self, mean, std):
        self.linear.weight.data /= std
        self.linear.bias.data -= mean
        self.linear.bias.data /= std

    def unnormalize(self, mean, std):
        self.linear.weight.data *= std
        self.linear.bias.data *= std
        self.linear.bias.data += mean