from typing import List, Callable

import torch
import torch.jit
import torch.nn as nn
from optfn.skip_connections import ResidualBlock
from rl_exp.noisy_linear import NoisyLinear

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


def create_fc(in_size: int, hidden_sizes: List[int], activation: Callable, norm: NormFactory = None, noisy_net=False):
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
        cls = NoisyLinear if noisy_net else Linear
        layer = [cls(n_in, n_out, bias=norm is None or not norm.disable_bias)]
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
            Linear(hidden_size, hidden_size),
            *norm(),
            nn.ReLU(),
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
        nn.ReLU(),
    )


class FCFeatureExtractor(FeatureExtractorBase):
    def __init__(self, input_size: int, hidden_sizes=(128, 128), activation=nn.Tanh, goal_size=0, noisy_net=False, **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.goal_size = goal_size
        self.model = create_fc(input_size, hidden_sizes, activation, self.norm_factory, noisy_net)
        self.out_embedding = Linear(goal_size, hidden_sizes[-1]) if goal_size != 0 else None

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

    def forward(self, input: torch.Tensor, logger=None, cur_step=None, goal=None, **kwargs):
        x = input.reshape(-1, input.shape[-1])
        if self.goal_size != 0:
            goal = goal.reshape(-1, goal.shape[-1])
        x = self._extract_features(x, goal, logger, cur_step)
        return x.reshape(*input.shape[:-1], -1)

    def _extract_features(self, x, goal,  logger, cur_step):
        for i, layer in enumerate(self.model):
            x = layer(x)
            if logger is not None:
                logger.add_histogram(f'layer_{i}_output', x, cur_step)
        if self.goal_size != 0:
            x = x * 2 * self.out_embedding(goal).sigmoid()
        return x


class BatchNormLast(nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-5, momentum=0.01, affine=True,
                 track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        return super().forward(input.reshape(-1, input.shape[-1])).reshape(input.shape)


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


class FCImaginationFeatureExtractor(FeatureExtractorBase):
    def __init__(self, input_size: int, hidden_sizes=(128, 128), activation=nn.Tanh, goal_size=None,
                 pd=None, num_sims=4, sim_depth=4, **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.goal_size = goal_size
        self.pd = pd
        self.num_sims = num_sims
        self.sim_depth = sim_depth
        self.gate_reduce = 8
        self.model = create_fc(input_size, hidden_sizes, activation, self.norm_factory)
        self.out_embedding = Linear(goal_size, hidden_sizes[-1]) if goal_size is not None else None

        h = hidden_sizes[-1]
        self.action_linear = Linear(pd.input_vector_len, h)
        self.vrld_layer = nn.Sequential(SiLU(), Linear(h, 3 + pd.prob_vector_len))
        self.feature_prepare_linear = Linear(h, h)
        self.world_model = nn.Sequential(
            ActivationNorm(1),
            SiLU(),
            Linear(h, h),
            ActivationNorm(1),
            SiLU(),
            Linear(h, h + h // self.gate_reduce),
        )
        self._end_gate_linear = Linear(h, h // self.gate_reduce)

    @property
    def output_size(self):
        return self.hidden_sizes[-1] * 2

    def forward(self, input: torch.Tensor, logger=None, cur_step=None, goal=None, **kwargs):
        x = input.reshape(-1, input.shape[-1])
        if self.goal_size is not None:
            goal = goal.reshape(-1, goal.shape[-1])

        x_hall, x = self._extract_features(x, goal, logger, cur_step)
        return dict(features=x_hall.reshape(*input.shape[:-1], -1), features_raw=x.reshape(*input.shape[:-1], -1))

    def _extract_features(self, x, goal,  logger, cur_step):
        for i, layer in enumerate(self.model):
            x = layer(x)
            if logger is not None:
                logger.add_histogram(f'layer_{i}_output', x, cur_step)

        if self.goal_size is not None:
            x = x * 2 * self.out_embedding(goal).sigmoid()

        x_fp = self.feature_prepare_linear(x.detach())
        x_hall = torch.cat([x, self._imagine(x_fp)], -1)
        return x_hall, x_fp

    def _imagine(self, features: Tensor) -> Tensor:
        features = torch.stack([features] * self.num_sims, 0)
        for _ in range(self.sim_depth):
            _, _, logits, _ = self._get_vrld(features)
            ac = self.pd.sample(logits.detach())
            features = self._world_model_step(features, ac)

        return silu(features.mean(0))

    def run_world_model(self, features, actions):
        data = [self._get_vrld(features)]
        for i, ac in enumerate(actions):
            features = self._world_model_step(features, ac)
            if i + 1 != len(actions):
                data.append(self._get_vrld(features))
        return [torch.stack(v, 0) for v in zip(*data)]

    def _world_model_step(self, features, action):
        h = self.hidden_sizes[-1]
        ac_enc = 2 * self.action_linear(self.pd.to_inputs(action)).sigmoid()
        new_f, gate = self.world_model(ac_enc * features).split([h, h // self.gate_reduce], -1)

        gate = gate.repeat_interleave(self.gate_reduce, -1).sigmoid()
        features = gate * new_f + (1 - gate) * features

        return features

    def _get_vrld(self, features):
        v, r, l, d = self.vrld_layer(features).split([1, 1, self.pd.prob_vector_len, 1], -1)
        return unsquash(v), r, unsquash(l), d


class FCActionFeatureExtractor(FeatureExtractorBase):
    def __init__(self, input_size: int, pd: ProbabilityDistribution, hidden_sizes=(256, 256), activation=nn.ReLU, **kwargs):
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


def create_ppo_fc_actor(observation_space, action_space, hidden_sizes=(128, 128),
                        activation=nn.Tanh, norm_factory: NormFactory=None,
                        split_policy_value_network=False, num_values=1, goal_size=0,
                        use_imagination=False):
    assert len(observation_space.shape) == 1

    fx_kwargs = dict(input_size=observation_space.shape[0], hidden_sizes=hidden_sizes, activation=activation,
                     norm_factory=norm_factory, goal_size=goal_size)

    if use_imagination:
        pd = make_pd(action_space)
        def fx_factory(): return FCImaginationFeatureExtractor(**fx_kwargs, pd=pd)
    else:
        def fx_factory(): return FCFeatureExtractor(**fx_kwargs)

    return create_ppo_actor(action_space, fx_factory, split_policy_value_network, num_values=num_values)


def create_impala_fc_actor(observation_space, action_space, hidden_sizes=(128, 128), activation=nn.Tanh,
                           norm_factory: NormFactory=None, num_values=1, goal_size=None, use_imagination=False,
                           split_policy_value_network=True, noisy_net=False):
    assert len(observation_space.shape) == 1

    fx_kwargs = dict(input_size=observation_space.shape[0], hidden_sizes=hidden_sizes, activation=activation,
                     norm_factory=norm_factory, goal_size=goal_size, noisy_net=noisy_net)

    if use_imagination:
        pd = make_pd(action_space)
        def fx_factory(): return FCImaginationFeatureExtractor(**fx_kwargs, pd=pd)
    else:
        def fx_factory(): return FCFeatureExtractor(**fx_kwargs)

    return create_impala_actor(action_space, fx_factory, split_policy_value_network, num_values, False)


def create_impala_attention_actor(observation_space, action_space, num_units, unit_size, hidden_size=256,
                                  activation=SiLU, num_values=1, goal_size=None, split_policy_value_network=True):
    assert len(observation_space.shape) == 1

    def fx_factory(): return FCAttentionFeatureExtractor(
        observation_space.shape[0], num_units, unit_size,
        hidden_size=hidden_size, activation=activation, goal_size=goal_size)

    return create_impala_actor(action_space, fx_factory, split_policy_value_network, num_values, False)


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
    head_q1 = ActionValueHead(fx_q1.output_size, pd=pd)
    head_q2 = ActionValueHead(fx_q2.output_size, pd=pd)
    models = {
        fx_policy: dict(logits=policy_head),
        fx_q1: dict(q1=head_q1),
        fx_q2: dict(q2=head_q2)
    }
    return ModularActor(models)