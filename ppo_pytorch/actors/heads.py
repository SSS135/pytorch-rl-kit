import torch.nn as nn
import torch
import torch.jit

from .utils import normalized_columns_initializer_
from ..common.probability_distributions import ProbabilityDistribution, CategoricalPd, BetaPd, DiagGaussianPd
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
        pass


class MultiActionValueHead(HeadBase):
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


class ActionValueHead(HeadBase):
    """
    Actor-critic head. Used in PPO / A3C.
    """

    def __init__(self, in_features, num_out=1, pd: ProbabilityDistribution = None):
        """
        Args:
            in_features: Input feature vector width.
            pd: Action probability distribution.
        """
        super().__init__(in_features)
        self.pd = pd
        self.num_out = num_out
        self.linear = Linear(in_features, num_out)
        self.action_enc = nn.Sequential(nn.Linear(pd.input_vector_len, in_features), nn.Sigmoid())

    def forward(self, x, actions=None, **kwargs):
        actions = self.pd.to_inputs(actions)
        q = self.linear(x * 2 * self.action_enc(actions))
        assert q.shape == (*actions.shape[:-1], self.num_out)
        return q

    def normalize(self, mean, std):
        self.linear.bias.data -= mean
        self.linear.bias.data /= std
        self.linear.weight.data /= std

    def unnormalize(self, mean, std):
        self.linear.weight.data *= std
        self.linear.bias.data *= std
        self.linear.bias.data += mean


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

    # def reset_weights(self):
    #     normalized_columns_initializer_(self.linear.weight.data, self.pd.init_column_norm)
    #     self.linear.bias.data.fill_(0)

    def forward(self, x, **kwargs):
        return self.linear(x)


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

    # def reset_weights(self):
    #     normalized_columns_initializer_(self.linear.weight.data, 1.0)
    #     self.linear.bias.data.fill_(0)

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
