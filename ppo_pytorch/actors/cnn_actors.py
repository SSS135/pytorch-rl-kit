import math
from functools import partial
from typing import Union, Callable

import torch
import torch.nn as nn
from optfn.skip_connections import ResidualBlock
from ppo_pytorch.actors.fc_actors import FCFeatureExtractor
from ppo_pytorch.actors.rnn_actors import RNNFeatureExtractor
from ppo_pytorch.common.probability_distributions import make_pd
from torch import autograd
from torch.autograd import Variable
from torch import Tensor

from .actors import FeatureExtractorBase, create_ppo_actor, create_impala_actor
from .norm_factory import NormFactory
from .utils import make_conv_heatmap, image_to_float, fixup_init
from ..common.make_grid import make_grid
from ppo_pytorch.common.activation_norm import ActivationNorm


class GroupTranspose(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, input):
        x = input.view(input.shape[0], self.groups, -1, *input.shape[2:])
        x = x.transpose(1, 2).contiguous()
        return x.view_as(input)


class ChannelShuffle(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.register_buffer('indices', torch.randperm(channels))

    def forward(self, input):
        return input[:, Variable(self.indices)].contiguous()


def log_policy_attention(states, head_out, logger, cur_step):
    states_grad = autograd.grad(
        head_out.logits.abs().mean() + head_out.state_values.abs().mean(), states,
        only_inputs=True, retain_graph=True)[0]
    with torch.no_grad():
        img = states_grad[:4]
        img.abs_()
        img /= img.view(4, -1).pow(2).mean(1).sqrt_().add_(1e-5).view(4, 1, 1, 1)
        img = img.view(-1, 1, *img.shape[2:]).abs()
        img = make_grid(img, 4, normalize=True, fill_value=0.1)
        logger.add_image('state_attention', img, cur_step)


def log_conv_filters(index: int, conv: nn.Conv2d, logger, cur_step):
    with torch.no_grad():
        channels = conv.in_channels * conv.out_channels
        shape = conv.weight.shape
        kernel_h, kernel_w = shape[2], shape[3]
        img = conv.weight.view(channels, 1, kernel_h, kernel_w).clone()
        max_img_size = 100 * 5
        img_size = channels * math.sqrt(kernel_h * kernel_w)
        if img_size > max_img_size:
            channels = channels * (max_img_size / img_size)
            channels = math.ceil(math.sqrt(channels)) ** 2
            img = img[:channels]
        img = make_conv_heatmap(img, scale=2 * img.std())
        img = make_grid(img, nrow=round(math.sqrt(channels)), normalize=False, fill_value=0.1)
        logger.add_image('conv_featrues_{}_img'.format(index), img, cur_step)
        logger.add_histogram('conv_features_{}_hist'.format(index), conv.weight, cur_step)


def log_conv_activations(index: int, x: torch.Tensor, logger, cur_step):
    with torch.no_grad():
        img = x[0].unsqueeze(1).clone()
        img = make_conv_heatmap(img)
        img = make_grid(img, nrow=round(math.sqrt(x.shape[1])), normalize=False, fill_value=0.1)
        logger.add_image('conv_activations_{}_img'.format(index), img, cur_step)
        logger.add_histogram('conv_activations_{}_hist'.format(index), x[0], cur_step)


class CNNFeatureExtractor(FeatureExtractorBase):
    """
    Convolution network.
    """
    def __init__(self, input_shape, cnn_kind: Union[str, Callable] = 'normal',
                 cnn_activation=partial(nn.ReLU, inplace=True),
                 add_positional_features=False, normalize_input=False, activation_norm=True, **kwargs):
        """
        Args:
            input_shape: Env's observation space
            cnn_kind: Type of cnn.
                'normal' - CNN from Nature DQN paper (Mnih et al. 2015)
                'custom' - largest CNN of custom structure
            cnn_activation: Activation function
        """
        super().__init__(**kwargs)
        self.input_shape = input_shape
        self.cnn_activation = cnn_activation
        self.cnn_kind = cnn_kind
        self.add_positional_features = add_positional_features
        self.normalize_input = normalize_input
        self.activation_norm = activation_norm
        self.convs = None
        self._prev_positions = None
        self._create_model()
        self._output_size = self._calc_linear_size()

    def _create_model(self):
        input_channels = self.input_shape[0] + (2 if self.add_positional_features else 0)
        # create convolutional layers
        if callable(self.cnn_kind):
            self.convs = self.cnn_kind(input_channels)
        elif self.cnn_kind == 'normal': # Nature DQN (1,683,456 parameters)
            self.convs = nn.ModuleList([
                self._make_cnn_layer(input_channels, 32, 8, 4, first_layer=True),
                self._make_cnn_layer(32, 64, 4, 2),
                self._make_cnn_layer(64, 64, 3, 1),
            ])
        elif self.cnn_kind == 'large': # custom (2,066,432 parameters)
            nf = 32
            self.convs = nn.ModuleList([
                self._make_cnn_layer(input_channels, nf, 4, 2, 0, first_layer=True),
                self._make_cnn_layer(nf, nf * 2, 4, 2, 0),
                self._make_cnn_layer(nf * 2, nf * 4, 4, 2, 1),
                self._make_cnn_layer(nf * 4, nf * 8, 4, 2, 1),
            ])
        elif self.cnn_kind == 'grouped': # custom grouped (6,950,912 parameters)
            nf = 32
            self.convs = nn.ModuleList([
                self._make_cnn_layer(input_channels, nf * 4, 4, 2, 0, first_layer=True),
                ChannelShuffle(nf * 4),
                self._make_cnn_layer(nf * 4, nf * 8, 4, 2, 0, groups=8),
                ChannelShuffle(nf * 8),
                self._make_cnn_layer(nf * 8, nf * 16, 4, 2, 1, groups=16),
                ChannelShuffle(nf * 16),
                self._make_cnn_layer(nf * 16, nf * 32, 4, 2, 1, groups=32),
                ChannelShuffle(nf * 32),
                self._make_cnn_layer(nf * 32, nf * 8, 3, 1, 1, groups=8),
            ])
        elif self.cnn_kind == 'impala':
            c_mult = 2
            def cnn_norm_fn(num_c):
                return (self.norm_factory.create_cnn_norm(num_c, False),) \
                    if self.norm_factory is not None and self.norm_factory.allow_cnn else ()
            def impala_block(c_in, c_out):
                return nn.Sequential(
                    nn.Conv2d(c_in, c_out, 3, 1, 1),
                    nn.MaxPool2d(3, 2, 1),
                    ResidualBlock(
                        *cnn_norm_fn(c_out),
                        nn.ReLU(True),
                        nn.Conv2d(c_out, c_out, 3, 1, 1),
                        *cnn_norm_fn(c_out),
                        nn.ReLU(True),
                        nn.Conv2d(c_out, c_out, 3, 1, 1),
                    ),
                    ResidualBlock(
                        *cnn_norm_fn(c_out),
                        nn.ReLU(True),
                        nn.Conv2d(c_out, c_out, 3, 1, 1),
                        *cnn_norm_fn(c_out),
                        nn.ReLU(True),
                        nn.Conv2d(c_out, c_out, 3, 1, 1),
                    ),
                )
            self.convs = nn.Sequential(
                impala_block(input_channels, 16 * c_mult),
                ActivationNorm(3),
                impala_block(16 * c_mult, 32 * c_mult),
                ActivationNorm(3),
                impala_block(32 * c_mult, 32 * c_mult),
                ActivationNorm(3),
                # impala_block(64 * c_mult, 64 * c_mult),
                nn.Sequential(
                    *cnn_norm_fn(32 * c_mult),
                    nn.ReLU(True),
                )
            )
        else:
            raise ValueError(self.cnn_kind)

    @property
    def output_size(self):
        return self._output_size

    def reset_weights(self):
        super().reset_weights()
        if self.cnn_kind == 'impala' or callable(self.cnn_kind):
            fixup_init(self.convs)

    def _calc_linear_size(self):
        with torch.no_grad():
            shape = 1, self.input_shape[0] + (2 if self.add_positional_features else 0), *self.input_shape[1:]
            out_shape = self._extract_features(torch.randn(shape)).shape
            return out_shape[1] * out_shape[2] * out_shape[3]

    def _make_cnn_layer(self, *args, first_layer=False, **kwargs):
        bias = self.norm_factory is None or not self.norm_factory.disable_bias or not self.norm_factory.allow_cnn
        return self._make_layer(nn.Conv2d(*args, **kwargs, bias=bias), first_layer=first_layer)

    def _make_layer(self, transf, first_layer=False):
        features = transf.out_channels

        # parts = [ActivationNormWrapper(transf)]
        parts = [transf]
        if self.activation_norm:
            parts.append(ActivationNorm(3))
        if self.norm_factory is not None and self.norm_factory.allow_cnn and \
                (self.norm_factory.allow_after_first_layer or not first_layer):
            parts.append(self.norm_factory.create_cnn_norm(features, first_layer))
        parts.append(self.cnn_activation())
        return nn.Sequential(*parts)

    def _extract_features(self, x, logger=None, cur_step=None):
        for i, layer in enumerate(self.convs):
            x = layer(x)
            if logger is not None:
                log_conv_activations(i, x, logger, cur_step)
        return x

    def _add_position_features(self, x: torch.Tensor):
        shape = list(x.shape)
        shape[-3] = 1
        if self._prev_positions is None or list(self._prev_positions.shape) != shape:
            h = torch.linspace(-1, 1, shape[-1], device=x.device).view(1, -1).expand(shape)
            v = torch.linspace(-1, 1, shape[-2], device=x.device).view(-1, 1).expand(shape)
            self._prev_positions = torch.cat([h, v], -3)
        return torch.cat([x, self._prev_positions], -3)

    def forward(self, input, logger=None, cur_step=None, **kwargs):
        input_shape = input.shape
        input = input.view(-1, *input_shape[-3:])

        input = image_to_float(input)
        if self.normalize_input:
            input = input * 2 - 1
        if self.add_positional_features:
            input = self._add_position_features(input)

        x = self._extract_features(input, logger, cur_step)
        x = x.view(x.size(0), -1)

        if logger is not None:
            logger.add_histogram('conv_activations_linear', x, cur_step)

        return x.view(*input_shape[:-3], x.shape[-1])


class CNNRNNFeatureExtractor(FeatureExtractorBase):
    def __init__(self, cnn_fx, rnn_fx):
        super().__init__()
        self.cnn_fx = cnn_fx
        self.rnn_fx = rnn_fx

    def reset_weights(self):
        self.cnn_fx.reset_weights()
        self.rnn_fx.reset_weights()

    @property
    def output_size(self):
        return self.rnn_fx.output_size

    def forward(self, input: Tensor, memory: Tensor, dones: Tensor, prev_rewards: Tensor, prev_actions: Tensor, **kwargs):
        features = self.cnn_fx(input, **kwargs)
        return self.rnn_fx(features, memory, dones, prev_rewards, prev_actions, **kwargs)


class CNNFCFeatureExtractor(FeatureExtractorBase):
    def __init__(self, cnn_fx, rnn_fx):
        super().__init__()
        self.cnn_fx = cnn_fx
        self.fc_fx = rnn_fx

    def reset_weights(self):
        self.cnn_fx.reset_weights()
        self.fc_fx.reset_weights()

    @property
    def output_size(self):
        return self.fc_fx.output_size

    def forward(self, input: Tensor, **kwargs):
        features = self.cnn_fx(input, **kwargs)
        return self.fc_fx(features, **kwargs)


def create_ppo_cnn_actor(observation_space, action_space, cnn_kind='normal',
                         cnn_activation=nn.ReLU, norm_factory: NormFactory=None,
                         split_policy_value_network=False, num_values=1,
                         add_positional_features=False, normalize_input=False):
    assert len(observation_space.shape) == 3

    def fx_factory(): return CNNFeatureExtractor(
        observation_space.shape, cnn_kind, cnn_activation, norm_factory=norm_factory,
        add_positional_features=add_positional_features, normalize_input=normalize_input)
    return create_ppo_actor(action_space, fx_factory, split_policy_value_network, num_values=num_values)


def create_impala_cnn_actor(observation_space, action_space, cnn_kind='normal',
                            activation=nn.ReLU, norm_factory: NormFactory=None, num_values=1,
                            add_positional_features=False, normalize_input=False, goal_size=0, fc_size=256,
                            split_policy_value_network=False):
    assert len(observation_space.shape) == 3

    def fx_factory():
        cnn_fx = CNNFeatureExtractor(
            observation_space.shape, cnn_kind, activation, norm_factory=norm_factory,
            add_positional_features=add_positional_features, normalize_input=normalize_input)
        fc_fx = FCFeatureExtractor(cnn_fx.output_size, (fc_size,), goal_size=goal_size, activation=activation)
        return CNNFCFeatureExtractor(cnn_fx, fc_fx)
    return create_impala_actor(action_space, fx_factory,
                               split_policy_value_network=split_policy_value_network, num_values=num_values, is_recurrent=False)


def create_impala_cnn_rnn_actor(observation_space, action_space, cnn_kind='normal',
                                cnn_activation=nn.ReLU, cnn_norm_factory: NormFactory=None,
                                rnn_layers=2, rnn_hidden_size=256,
                                num_values=1, add_positional_features=False, normalize_input=False, goal_size=0):
    assert len(observation_space.shape) == 3
    pd = make_pd(action_space)
    def fx_factory():
        cnn_fx = CNNFeatureExtractor(
            observation_space.shape, cnn_kind, cnn_activation, norm_factory=cnn_norm_factory,
            add_positional_features=add_positional_features, normalize_input=normalize_input)
        rnn_fx = RNNFeatureExtractor(pd, cnn_fx.output_size, rnn_hidden_size, rnn_layers, env_input=False)
        return CNNRNNFeatureExtractor(cnn_fx, rnn_fx)
    return create_impala_actor(
        action_space, fx_factory, split_policy_value_network=False, num_values=num_values, is_recurrent=True)