from typing import List, Callable

import torch.nn as nn

from .actors import FeatureExtractorBase, ModularActor, create_ppo_actor
from .heads import StateValueQuantileHead, PolicyHead, StateValueHead
from .norm_factory import NormFactory
from ..common.probability_distributions import make_pd
from optfn.skip_connections import ResidualBlock


def create_fc(in_size: int, hidden_sizes: List[int], activation: Callable, norm: NormFactory = None):
    """
    Create fully connected network
    Args:
        in_size: Input vector size.
        out_size: Optional. Output vector size. Additional layer is appended if not None.
        hidden_sizes: Width of hidden layers.
        activation: Activation function
        norm: Used normalization technique

    Returns: `nn.Sequential` of layers. Each layer is also `nn.Sequential` containing (linear, [norm], activation).
        If `out_size` is not None, last layer is just linear transformation, without norm or activation.

    """
    seq = []
    for i in range(len(hidden_sizes)):
        n_in = in_size if i == 0 else hidden_sizes[i - 1]
        n_out = hidden_sizes[i]
        layer = [nn.Linear(n_in, n_out, bias=norm is None or not norm.disable_bias)]
        if norm is not None and norm.allow_fc and (norm.allow_after_first_layer or i != 0):
            layer.append(norm.create_fc_norm(n_out, i == 0))
        layer.append(activation())
        seq.append(nn.Sequential(*layer))
    seq = nn.Sequential(*seq)
    return seq


class FCFeatureExtractor(FeatureExtractorBase):
    def __init__(self, input_size: int, hidden_sizes=(128, 128), activation=nn.Tanh, **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.model = create_fc(input_size, hidden_sizes, activation, self.norm_factory)

    @property
    def output_size(self):
        return self.hidden_sizes[-1]

    def forward(self, input, logger=None, cur_step=None, **kwargs):
        x = input
        for i, layer in enumerate(self.model):
            x = layer(x)
            if logger is not None:
                logger.add_histogram(f'layer {i} output', x, cur_step)
        return x


def create_ppo_fc_actor(observation_space, action_space, hidden_sizes=(128, 128),
                        activation=nn.Tanh, norm_factory: NormFactory=None,
                        iqn=False, split_policy_value_network=True, num_bins=1):
    assert len(observation_space.shape) == 1

    def fx_factory(): return FCFeatureExtractor(
        observation_space.shape[0], hidden_sizes, activation, norm_factory=norm_factory)
    return create_ppo_actor(action_space, fx_factory, iqn, split_policy_value_network, num_bins=num_bins)