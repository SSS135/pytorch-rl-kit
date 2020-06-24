import torch
import torch.nn as nn
from ppo_pytorch.common.silu import SiLU

from .utils import normalized_columns_initializer_
from ..common.probability_distributions import ProbabilityDistribution, CategoricalPd
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


class PolicyHead(HeadBase):
    """
    Actor-critic head. Used in PPO / A3C.
    """

    def __init__(self, in_features, pd: ProbabilityDistribution, layer_norm=False):
        """
        Args:
            in_features: Input feature vector width.
            pd: Action probability distribution.
        """
        super().__init__(in_features)
        self.pd = pd
        self.layer_norm = layer_norm
        self.linear = Linear(in_features, self.pd.prob_vector_len)
        self.ln = nn.LayerNorm(in_features)
        self.reset_weights()

    def reset_weights(self):
        if self.layer_norm:
            normalized_columns_initializer_(self.linear.weight.data, 1.0)
            self.linear.bias.data.fill_(0)
        else:
            normalized_columns_initializer_(self.linear.weight.data, 0.1)
            self.linear.bias.data.fill_(0)

    def forward(self, x, **kwargs):
        if self.layer_norm:
            x = self.ln(x)
        return self.linear(x)


class ActionValueHead(HeadBase):
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
        self.action_enc = nn.Sequential(
            Linear(pd.input_vector_len, 128),
            nn.Tanh(),
            Linear(128, in_features * 2),
        )

    def reset_weights(self):
        normalized_columns_initializer_(self.linear.weight.data, 1.0)
        self.linear.bias.data.fill_(0)

    def forward(self, x, actions=None, action_noise_scale=0, **kwargs):
        actions = self.pd.to_inputs(actions)
        if action_noise_scale != 0:
            actions = actions + action_noise_scale * torch.randn_like(actions)
        ac_add, ac_mul = self.action_enc(actions).chunk(2, -1)
        q = self.linear(x * ac_mul + ac_add)
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


class DoubleActionValueHead(HeadBase):
    def __init__(self, in_features, num_out=1, pd: ProbabilityDistribution = None):
        """
        Args:
            in_features: Input feature vector width.
            pd: Action probability distribution.
        """
        super().__init__(in_features)
        self.pd = pd
        self.num_out = num_out
        self.linears = nn.ModuleList([Linear(in_features // 2, num_out) for _ in range(2)])
        self.action_encs = nn.ModuleList([nn.Sequential(
            Linear(pd.input_vector_len, 128),
            nn.Tanh(),
            Linear(128, in_features),
        ) for _ in range(2)])

    def reset_weights(self):
        for linear in self.linears:
            normalized_columns_initializer_(linear.weight.data, 1.0)
            linear.bias.data.fill_(0)

    def forward(self, features, actions=None, action_noise_scale=0, **kwargs):
        actions = self.pd.to_inputs(actions)
        if action_noise_scale != 0:
            actions = actions + action_noise_scale * torch.randn_like(actions)
        ac_enc = [enc(actions).chunk(2, -1) for enc in self.action_encs]
        qs = [linear(x * ac_mul + ac_add)
              for linear, x, (ac_mul, ac_add) in zip(self.linears, features.chunk(2, -1), ac_enc)]
        q = torch.cat(qs, -1)
        assert q.shape == (*actions.shape[:-1], 2 * self.num_out)
        return q

    def normalize(self, mean, std):
        for linear in self.linears:
            linear.bias.data -= mean
            linear.bias.data /= std
            linear.weight.data /= std


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
