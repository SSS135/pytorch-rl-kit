from collections import OrderedDict
from typing import List, Callable

import torch
import torch.jit
import torch.nn as nn
from optfn.skip_connections import ResidualBlock

from ppo_pytorch.common.squash import unsquash
from ppo_pytorch.actors.transformer import TrPriorFirstLayer, SimpleTrLayer
from ppo_pytorch.common.silu import SiLU, silu
from torch import Tensor

from .actors import FeatureExtractorBase, ModularActor, create_ppo_actor, create_impala_actor
from .heads import PolicyHead, StateValueHead, ActionValueHead
from .norm_factory import NormFactory
from ..common.activation_norm import ActivationNorm
from ..common.probability_distributions import ProbabilityDistribution, make_pd
from ..config import Linear
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


Act = nn.SiLU


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
        n_out = hidden_sizes[i] * (2 if activation == SwiGLU else 1)
        layer = [Linear(n_in, n_out, bias=norm is None or not norm.disable_bias)]
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
            Act(),
            Linear(hidden_size, hidden_size),
            *norm(),
            Act(),
            Linear(hidden_size, hidden_size),
        )
    return nn.Sequential(
        Linear(input_size, hidden_size),
        res_block(),
        res_block(),
        res_block(),
        res_block(),
        res_block(),
        res_block(),
        *norm(),
        Act(),
    )


class FCFeatureExtractor(FeatureExtractorBase):
    def __init__(self, input_size: int, hidden_sizes=(128, 128), activation=Act, **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.model = create_fc(input_size, hidden_sizes, activation, self.norm_factory)
        self.model = torch.compile(self.model, fullgraph=True, mode='max-autotune')

    @property
    def output_size(self):
        return self.hidden_sizes[-1]

    def forward(self, input: torch.Tensor, logger=None, cur_step=None, **kwargs):
        x = input.reshape(-1, input.shape[-1])
        x = self._extract_features(x, logger, cur_step)
        return x.reshape(*input.shape[:-1], -1)

    def _extract_features(self, x, logger, cur_step):
        return self.model(x)
        # for i, layer in enumerate(self.model):
        #     x = layer(x)
        #     if logger is not None:
        #         logger.add_histogram(f'layer_{i}_output', x, cur_step)
        # return x


class GroupNormLast(nn.GroupNorm):
    def forward(self, input):
        return super().forward(input.reshape(-1, input.shape[-1])).reshape(input.shape)


class FCAttentionFeatureExtractor(FeatureExtractorBase):
    def __init__(self, input_size: int, num_units: int, unit_size: int,
                 hidden_size=256, activation=SiLU, goal_size=None,
                 num_full_layers=0, num_simple_layers=2, **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.num_units = num_units
        self.unit_size = unit_size
        self.hidden_size = hidden_size
        self.activation = activation
        self.goal_size = goal_size
        self.num_full_layers = num_full_layers
        self.num_simple_layers = num_simple_layers

        self.personal_fc = nn.Sequential(
            Linear(input_size - num_units * unit_size, hidden_size),
            activation(),
            Linear(hidden_size, hidden_size),
            activation(),
        )
        self.unit_fc = nn.Sequential(
            Linear(unit_size, hidden_size),
            activation(),
            Linear(hidden_size, hidden_size),
            activation(),
        )
        self.end_fc = nn.Sequential(
            nn.LayerNorm(hidden_size),
            Linear(hidden_size, hidden_size),
            activation(),
        )
        self.full_tr_layers = nn.ModuleList([TrPriorFirstLayer(256, 32, 256 // 32) for _ in range(self.num_full_layers)])
        self.simple_tr_layers = nn.ModuleList([SimpleTrLayer(256, 32, 256 // 32) for _ in range(self.num_simple_layers)])
        self.out_embedding = Linear(goal_size, hidden_size) if goal_size is not None else None

    @property
    def output_size(self):
        return self.hidden_size

    def forward(self, input: torch.Tensor, logger=None, cur_step=None, goal=None, **kwargs):
        x = input.reshape(-1, input.shape[-1])
        if self.goal_size is not None:
            goal = goal.reshape(-1, goal.shape[-1])
        NU, US = self.num_units, self.unit_size
        units, x = x[..., :NU * US].reshape(*x.shape[:-1], NU, US), x[..., NU * US:]
        x = self._extract_features(x, units, goal, logger, cur_step)
        return x.reshape(*input.shape[:-1], -1)

    def _extract_features(self, personal, units, goal, logger, cur_step):
        units = self.unit_fc(units)
        x = self.personal_fc(personal)

        B, NU, H = units.shape
        assert x.shape == (B, H)

        if self.num_full_layers > 0:
            x = torch.cat([x.unsqueeze(1), units], 1)
            for layer in self.full_tr_layers:
                x = layer(x)

        if self.num_simple_layers > 0 and self.num_full_layers > 0:
            x, units = x[:, 0, :], x[:, 1:, :]
        for layer in self.simple_tr_layers:
            x = layer(x, units)

        x = self.end_fc(x)
        # if self.goal_size is not None:
        #     x = x * 2 * self.out_embedding(goal).sigmoid()
        return x


class FCActionFeatureExtractor(FeatureExtractorBase):
    def __init__(self, input_size: int, pd: ProbabilityDistribution, hidden_sizes=(256, 256), activation=Act, **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.pd = pd
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.model = self._create_fc()
        self.ac_encoder = nn.Sequential(
            Linear(pd.input_vector_len, 128),
            activation(),
            Linear(128, 2 * sum(hidden_sizes))
        )

    @property
    def output_size(self):
        return self.hidden_sizes[-1]

    def forward(self, input: torch.Tensor, logger=None, cur_step=None, actions=None, **kwargs):
        x = input.view(-1, input.shape[-1])
        ac_inputs = self.pd.to_inputs(actions).view(-1, actions.shape[-1])
        ac_inputs = self.ac_encoder(ac_inputs).split(self.hidden_sizes * 2, -1)

        for i, (layer, ac_mul, ac_add) in enumerate(zip(self.model, ac_inputs[:len(self.hidden_sizes)], ac_inputs[len(self.hidden_sizes):])):
            x = layer(x)
            x = x * ac_mul + ac_add
            if logger is not None:
                logger.add_histogram(f'layer_{i}_output', x, cur_step)
        return x.view(*input.shape[:-1], x.shape[-1])

    def _create_fc(self):
        norm = self.norm_factory
        seq = []
        for i in range(len(self.hidden_sizes)):
            n_in = self.input_size if i == 0 else self.hidden_sizes[i - 1]
            n_out = self.hidden_sizes[i]
            layer = [Linear(n_in, n_out, bias=norm is None or not norm.disable_bias)]
            if norm is not None and norm.allow_fc and (norm.allow_after_first_layer or i != 0):
                layer.append(norm.create_fc_norm(n_out, i == 0))
            layer.append(self.activation())
            seq.append(nn.Sequential(*layer))
        return nn.Sequential(*seq)


def create_ppo_fc_actor(observation_space, action_space, hidden_sizes=(64, 64),
                        activation=Act, norm_factory: NormFactory=None,
                        split_policy_value_network=True, num_values=1):
    assert len(observation_space.shape) == 1

    fx_kwargs = dict(input_size=observation_space.shape[0], hidden_sizes=hidden_sizes, activation=activation,
                     norm_factory=norm_factory)

    def fx_factory(): return FCFeatureExtractor(**fx_kwargs)

    return create_ppo_actor(action_space, fx_factory, split_policy_value_network, num_values=num_values)


def create_advppo_fc_actor(observation_space, action_space, hidden_sizes=(128, 128),
                           activation=Act, norm_factory: NormFactory=None, num_values=1):
    assert len(observation_space.shape) == 1

    fx_kwargs = dict(hidden_sizes=hidden_sizes, activation=activation, norm_factory=norm_factory)

    def fx_rand_factory(): return FCFeatureExtractor(input_size=2 * observation_space.shape[0], **fx_kwargs)
    def fx_action_factory(): return FCActionFeatureExtractor(input_size=observation_space.shape[0], pd=pd, **fx_kwargs)

    pd = make_pd(action_space)
    fx_disc, fx_gen, fx_q = fx_action_factory(), fx_rand_factory(), fx_action_factory()

    disc_head = StateValueHead(fx_disc.output_size, num_out=num_values)
    gen_head = PolicyHead(fx_gen.output_size, pd=pd)
    q_head = StateValueHead(fx_q.output_size, num_out=num_values)
    models = OrderedDict([
        (fx_disc, dict(disc=disc_head)),
        (fx_gen, dict(gen=gen_head)),
        (fx_q, dict(q=q_head)),
    ])
    return ModularActor(models, False)


def create_impala_fc_actor(observation_space, action_space, hidden_sizes=(128, 128), activation=Act,
                           norm_factory: NormFactory=None, num_values=1, split_policy_value_network=True):
    assert len(observation_space.shape) == 1

    fx_kwargs = dict(input_size=observation_space.shape[0], hidden_sizes=hidden_sizes, activation=activation,
                     norm_factory=norm_factory)

    def fx_factory(): return FCFeatureExtractor(**fx_kwargs)

    return create_impala_actor(action_space, fx_factory, split_policy_value_network, num_values, False)


def create_impala_attention_actor(observation_space, action_space, num_units, unit_size, hidden_size=256,
                                  activation=SiLU, num_values=1, goal_size=None, split_policy_value_network=True):
    assert len(observation_space.shape) == 1

    def fx_factory(): return FCAttentionFeatureExtractor(
        observation_space.shape[0], num_units, unit_size,
        hidden_size=hidden_size, activation=activation, goal_size=goal_size)

    return create_impala_actor(action_space, fx_factory, split_policy_value_network, num_values, False)


def create_sac_fc_actor(observation_space, action_space, hidden_sizes=(128, 128), activation=Act,
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