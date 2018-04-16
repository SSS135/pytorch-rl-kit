import torch.nn as nn
import torch.nn.init as init

from .actors import ActorOutput
from ..common.probability_distributions import ProbabilityDistribution, CategoricalPd
from optfn.grad_running_norm import GradRunningNorm


class HeadBase(nn.Module):
    """
    Base class for output of `Actor`. Different heads used for different learning algorithms.
    """

    def __init__(self, in_features, pd: ProbabilityDistribution):
        """
        Args:
            in_features: Input feature vector width.
            pd: Action probability distribution.
        """
        super().__init__()
        self.in_features = in_features
        self.pd = pd

    def reset_weights(self):
        raise NotImplementedError()


class ActionValuesHead(HeadBase):
    """
    Head with state-values. Used in Q learning. Support dueling architecture.
    """

    def __init__(self, in_features, pd: CategoricalPd, dueling=True):
        """
        Args:
            in_features: Input feature vector width.
            pd: Action probability distribution.
            dueling: Use dueling architecture.
        """
        super().__init__(in_features, pd)
        assert isinstance(pd, CategoricalPd)
        self.dueling = dueling
        self.linear = nn.Linear(in_features, self.pd.prob_vector_len + 1)

    def reset_weights(self):
        init.orthogonal(self.linear.weight.data, 1)
        # normalized_columns_initializer(self.linear.weight.data, 1.0)
        self.linear.bias.data.fill_(0)

    def forward(self, x):
        av = self.linear(x)
        ac = self.pd.prob_vector_len
        A = av[:, :ac].contiguous()
        V = av[:, ac:].contiguous()
        if self.dueling:
            Q = V + (A - A.max(1, keepdim=True)[0])
        else:
            Q = A
        return ActorOutput(action_values=Q, head_raw=av)


class ActorCriticHead(HeadBase):
    """
    Actor-critic head. Used in PPO / A3C.
    """

    def __init__(self, in_features, pd: ProbabilityDistribution):
        """
        Args:
            in_features: Input feature vector width.
            pd: Action probability distribution.
        """
        super().__init__(in_features, pd)
        self.linear = nn.Linear(in_features, self.pd.prob_vector_len + 1)

    def reset_weights(self):
        init.orthogonal(self.linear.weight.data[0:1], 1)
        init.orthogonal(self.linear.weight.data[1:], 0.01)
        # normalized_columns_initializer(self.linear.weight.data[0:1], 1.0)
        # normalized_columns_initializer(self.linear.weight.data[1:], 0.01)
        self.linear.bias.data.fill_(0)

    def forward(self, x):
        x = self.linear(x)
        values = x[..., 0]
        probs = x[..., 1:]
        return ActorOutput(probs=probs, state_values=values, head_raw=x)
