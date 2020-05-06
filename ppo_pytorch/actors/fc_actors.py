import math
from typing import List, Callable

import torch.nn as nn
from ppo_pytorch.common.activation_norm import ActivationNorm

from .actors import FeatureExtractorBase, ModularActor, create_ppo_actor
from .heads import PolicyHead, StateValueHead
from .norm_factory import NormFactory
from ..common.probability_distributions import LinearTanhPd, ProbabilityDistribution, make_pd
import torch
from ..config import Linear
from optfn.skip_connections import ResidualBlock
import torch.jit


def create_fc(in_size: int, hidden_sizes: List[int], activation: Callable, norm: NormFactory = None, activation_norm=False):
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
        layer = [Linear(n_in, n_out, bias=norm is None or not norm.disable_bias)]
        if activation_norm:
            layer.append(ActivationNorm())
        if norm is not None and norm.allow_fc and (norm.allow_after_first_layer or i != 0):
            layer.append(norm.create_fc_norm(n_out, i == 0))
        layer.append(activation())
        seq.append(nn.Sequential(*layer))
    seq = nn.Sequential(*seq)
    return seq


def create_residual_fc(input_size, hidden_size, use_norm=False):
    def norm():
        return (nn.LayerNorm(hidden_size),) if use_norm else ()
    def res_block():
        return ResidualBlock(
            *norm(),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            *norm(),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        res_block(),
        res_block(),
        res_block(),
        res_block(),
        res_block(),
        res_block(),
        *norm(),
        nn.ReLU(),
    )


class FCFeatureExtractor(FeatureExtractorBase):
    def __init__(self, input_size: int, hidden_sizes=(128, 128), activation=nn.Tanh, **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.model = create_fc(input_size, hidden_sizes, activation, self.norm_factory)
        # self.model = create_residual_fc(input_size, hidden_sizes[0])
        # super().reset_weights()
        # fixup_init(self.model)
        # self.model = torch.jit.trace_module(self.model, dict(forward=torch.randn((8, input_size))))

    # def reset_weights(self):
    #     pass
    #     # super().reset_weights()
    #     # fixup_init(self.model)

    @property
    def output_size(self):
        return self.hidden_sizes[-1]

    def forward(self, input: torch.Tensor, logger=None, cur_step=None, **kwargs):
        x = input.reshape(-1, input.shape[-1])
        # x = self.model(x)
        for i, layer in enumerate(self.model):
            x = layer(x)
            if logger is not None:
                logger.add_histogram(f'layer_{i}_output', x, cur_step)
        return x.reshape(*input.shape[:-1], -1)


class FCActionFeatureExtractor(FeatureExtractorBase):
    def __init__(self, input_size: int, pd: ProbabilityDistribution, hidden_sizes=(256, 256), activation=nn.ReLU, **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.pd = pd
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.model = self._create_fc()

    @property
    def output_size(self):
        return self.hidden_sizes[-1]

    def forward(self, input: torch.Tensor, logger=None, cur_step=None, actions=None, logits=None, **kwargs):
        x = input.view(-1, input.shape[-1])
        ac_inputs = self.pd.to_inputs(actions)
        ac_inputs = (ac_inputs + 0.05 * torch.randn_like(ac_inputs)).view(-1, ac_inputs.shape[-1])
        # ac_inputs = logits.detach().view(-1, logits.shape[-1])
        for i, layer in enumerate(self.model):
            x = layer(torch.cat([x, ac_inputs], -1))
            if logger is not None:
                logger.add_histogram(f'layer_{i}_output', x, cur_step)
        return x.view(*input.shape[:-1], x.shape[-1])

    def _create_fc(self):
        norm = self.norm_factory
        seq = []
        for i in range(len(self.hidden_sizes)):
            n_in = self.pd.input_vector_len + (self.input_size if i == 0 else self.hidden_sizes[i - 1])
            n_out = self.hidden_sizes[i]
            layer = [Linear(n_in, n_out, bias=norm is None or not norm.disable_bias)]
            if norm is not None and norm.allow_fc and (norm.allow_after_first_layer or i != 0):
                layer.append(norm.create_fc_norm(n_out, i == 0))
            layer.append(self.activation())
            seq.append(nn.Sequential(*layer))
        return nn.Sequential(*seq)


def create_ppo_fc_actor(observation_space, action_space, hidden_sizes=(128, 128),
                        activation=nn.Tanh, norm_factory: NormFactory=None,
                        split_policy_value_network=True, num_values=1):
    assert len(observation_space.shape) == 1

    def fx_factory(): return FCFeatureExtractor(
        observation_space.shape[0], hidden_sizes, activation, norm_factory=norm_factory)
    return create_ppo_actor(action_space, fx_factory, split_policy_value_network, num_out=num_values)


def create_sac_fc_actor(observation_space, action_space, hidden_sizes=(256, 256), activation=nn.ReLU,
                        norm_factory: NormFactory = None):
    assert len(observation_space.shape) == 1
    pd = make_pd(action_space)

    def fx_policy_factory(): return FCFeatureExtractor(
        observation_space.shape[0], hidden_sizes, activation, norm_factory=norm_factory)

    def fx_q_factory(): return FCActionFeatureExtractor(
        observation_space.shape[0], pd, hidden_sizes, activation, norm_factory=norm_factory)

    fx_policy, fx_q1, fx_q2 = fx_policy_factory(), fx_q_factory(), fx_q_factory()

    policy_head = PolicyHead(fx_policy.output_size, pd=pd)
    head_q1 = StateValueHead(fx_q1.output_size, pd=pd)
    head_q2 = StateValueHead(fx_q2.output_size, pd=pd)
    models = {
        fx_policy: dict(logits=policy_head),
        fx_q1: dict(q1=head_q1),
        fx_q2: dict(q2=head_q2)
    }
    return ModularActor(models)