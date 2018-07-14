import math
from functools import partial
from typing import Optional, List, Callable

import gym.spaces
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from optfn.switchable_norm import SwitchableNorm2d, SwitchableNorm1d
from optfn.auto_norm import AutoNorm
from torch import autograd
from torch.autograd import Variable

from .heads import HeadOutput
from .utils import weights_init, make_conv_heatmap, image_to_float
from ..common.make_grid import make_grid
from ..common.probability_distributions import make_pd


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


class Actor(nn.Module):
    """
    Base class for network in reinforcement learning algorithms.
    """

    def __init__(self, observation_space: gym.Space, action_space: gym.Space, norm: str = None,
                 weight_init=init.xavier_uniform_, weight_init_gain=math.sqrt(2)):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.weight_init_gain = weight_init_gain
        self.weight_init = weight_init
        self.norm = norm
        self.hidden_code_size = None

        self.do_log = False
        self.logger = None
        self._step = 0
        self.pd = make_pd(action_space)

    def set_log(self, logger, do_log: bool, step: int):
        """
        Set logging state.
        Args:
            logger: Logger object.
            do_log: Enable logging
            step: Current training step
        """
        self.logger = logger
        self.do_log = do_log
        self._step = step

    def reset_weights(self):
        self.apply(partial(weights_init, init_alg=self.weight_init, gain=self.weight_init_gain))
        for m in self.modules():
            if m is not self and hasattr(m, 'reset_weights'):
                m.reset_weights()

    @staticmethod
    def create_fc(in_size: int, out_size: Optional[int], hidden_sizes: List[int],
                  activation: Callable, norm: str = None):
        """
        Create fully connected network
        Args:
            in_size: Input vector size.
            out_size: Optional. Output vector size. Additional layer is appended if not None.
            hidden_sizes: Width of hidden layers.
            activation: Activation function
            norm: Used normalization technique.
                None - no normalization
                'layer' - layer normalization
                'batch' - batch normalization
                'group' - group normalization

        Returns: `nn.Sequential` of layers. Each layer is also `nn.Sequential` containing (linear, [norm], activation).
            If `out_size` is not None, last layer is just linear transformation, without norm or activation.

        """
        assert norm in (None, 'layer', 'batch', 'group')
        seq = []
        for i in range(len(hidden_sizes)):
            n_in = in_size if i == 0 else hidden_sizes[i - 1]
            n_out = hidden_sizes[i]
            layer = [nn.Linear(n_in, n_out)]
            if i != 0:
                if norm == 'group':
                    layer.append(nn.GroupNorm(n_out // 16, n_out))
                elif norm == 'layer':
                    layer.append(nn.LayerNorm(n_out))
                elif norm == 'batch':
                    layer.append(nn.BatchNorm1d(n_out, momentum=0.01))
            layer.append(activation())
            seq.append(nn.Sequential(*layer))
        if out_size is not None:
            seq.append(nn.Linear(hidden_sizes[-1], out_size))
        seq = nn.Sequential(*seq)
        return seq

    def log_conv_activations(self, index: int, x: Variable):
        with torch.no_grad():
            img = x[0].unsqueeze(1).clone()
            img = make_conv_heatmap(img)
            img = make_grid(img, nrow=round(math.sqrt(x.shape[1])), normalize=False, fill_value=0.1)
            self.logger.add_image('conv activations {} img'.format(index), img, self._step)
            self.logger.add_histogram('conv activations {} hist'.format(index), x[0], self._step)

    def log_conv_filters(self, index: int, conv: nn.Conv2d):
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
            self.logger.add_image('conv featrues {} img'.format(index), img, self._step)
            self.logger.add_histogram('conv features {} hist'.format(index), conv.weight, self._step)

    def log_policy_attention(self, states, head_out):
        states_grad = autograd.grad(
            head_out.probs.abs().mean() + head_out.state_values.abs().mean(), states,
            only_inputs=True, retain_graph=True)[0]
        with torch.no_grad():
            img = states_grad[:4]
            img.abs_()
            img /= img.view(4, -1).pow(2).mean(1).sqrt_().add_(1e-5).view(4, 1, 1, 1)
            img = img.view(-1, 1, *img.shape[2:]).abs()
            # img = make_conv_heatmap(img, scale=2*img.std())
            img = make_grid(img, 4, normalize=True, fill_value=0.1)
            self.logger.add_image('state attention', img, self._step)


class FCActor(Actor):
    """
    Fully connected network.
    """

    def __init__(self, observation_space: gym.Space, action_space: gym.Space, head_factory: Callable,
                 hidden_sizes=(128, 128), activation=nn.Tanh, hidden_code_type='input', dual_net=False, **kwargs):
        """
        Args:
            observation_space: Env's observation space
            action_space: Env's action space
            head_factory: Function which accept (hidden vector size, `ProbabilityDistribution`) and return `HeadBase`
            hidden_sizes: List of hidden layers sizes
            activation: Activation function
        """
        super().__init__(observation_space, action_space, **kwargs)
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.hidden_code_type = hidden_code_type
        self.dual_net = dual_net

        obs_len = int(np.product(observation_space.shape))

        if dual_net:
            self.value_net = FCActor(observation_space, action_space, head_factory,
                                     hidden_sizes, activation, hidden_code_type, dual_net=False, **kwargs)
        else:
            self.value_net = None

        self.hidden_code_size = obs_len if hidden_code_type == 'input' else \
            (self.pd.prob_vector_len if hidden_code_type == 'probs' else hidden_sizes[-1])
        self.linear = self.create_fc(obs_len, None, hidden_sizes, activation, self.norm)
        self.head = head_factory(hidden_sizes[-1], self.pd)
        self.reset_weights()

    def reset_weights(self):
        super().reset_weights()
        self.head.reset_weights()

    def forward(self, input, hidden_code_input=False, only_hidden_code_output=False):
        if hidden_code_input and only_hidden_code_output:
            return HeadOutput(hidden_code=input)

        # hidden_code = input if self.input_as_hidden_code or hidden_code_input else self.linear[0][:-1](input)
        # if only_hidden_code_output:
        #     return HeadOutput(hidden_code=hidden_code)

        x = input
        # if hidden_code_input:
        #     hidden_code = x
        #     x = self.linear[-1][-1](x)
        # else:
        hidden_code = None
        if self.hidden_code_type == 'last':
            if hidden_code_input:
                hidden_code = input
                x = self.linear[-1][-1](input)
            else:
                for i, layer in enumerate(self.linear):
                    if i + 1 == len(self.linear):
                        hidden_code = layer[:-1](x)
                        if only_hidden_code_output:
                            return HeadOutput(hidden_code=hidden_code)
                        x = layer[-1](hidden_code)
                    else:
                        x = layer(x)
                    if self.do_log:
                        self.logger.add_histogram(f'layer {i} output', x, self._step)
        elif self.hidden_code_type == 'probs' and hidden_code_input:
            return HeadOutput(hidden_code=input, probs=input)
        else:
            for i, layer in enumerate(self.linear):
                # x = layer(x) if self.input_as_hidden_code or i != 0 else layer[-1](x)
                if i == 0 and self.hidden_code_type != 'probs': # i + 1 == len(self.linear):
                    hidden_code = x if hidden_code_input and self.hidden_code_type != 'input' else layer[:-1](x)
                    if only_hidden_code_output:
                        return HeadOutput(hidden_code=input if self.hidden_code_type == 'input' else hidden_code)
                    x = layer[-1](hidden_code)
                else:
                    x = layer(x)
                if self.do_log:
                    self.logger.add_histogram(f'layer {i} output', x, self._step)

        head = self.head(x)
        if self.hidden_code_type == 'input':
            hidden_code = input
        elif self.hidden_code_type == 'probs':
            hidden_code = head.probs
        head.hidden_code = hidden_code
        if self.dual_net:
            head.state_values = self.value_net(input, hidden_code_input=hidden_code_input).state_values
        return head


class CNNActor(Actor):
    """
    Convolution network.
    """
    def __init__(self, observation_space, action_space, head_factory, cnn_kind='large',
                 cnn_activation=nn.ReLU, linear_activation=nn.ReLU, cnn_hidden_code=False, **kwargs):
        """
        Args:
            observation_space: Env's observation space
            action_space: Env's action space
            head_factory: Function which accept (hidden vector size, `ProbabilityDistribution`) and return `HeadBase`
            cnn_kind: Type of cnn.
                'small' - small CNN from arxiv DQN paper (Mnih et al. 2013)
                'large' - bigger CNN from Nature DQN paper (Mnih et al. 2015)
                'custom' - largest CNN of custom structure
            cnn_activation: Activation function
        """
        super().__init__(observation_space, action_space, **kwargs)
        self.cnn_activation = cnn_activation
        self.linear_activation = linear_activation
        self.cnn_kind = cnn_kind
        self.head_factory = head_factory
        self.cnn_hidden_code = cnn_hidden_code

        # create convolutional layers
        switch_norm = self.norm is not None and ('switch' in self.norm or 'auto' in self.norm)
        if cnn_kind == 'normal': # Nature DQN (1,683,456 parameters)
            self.convs = nn.ModuleList([
                self.make_layer(nn.Conv2d(observation_space.shape[0], 32, 8, 4), allow_norm=switch_norm),
                self.make_layer(nn.Conv2d(32, 64, 4, 2)),
                self.make_layer(nn.Conv2d(64, 64, 3, 1)),
            ])
            self.linear = self.make_layer(nn.Linear(3136, 512))
        elif cnn_kind == 'large': # custom (2,066,432 parameters)
            nf = 32
            self.convs = nn.ModuleList([
                self.make_layer(nn.Conv2d(observation_space.shape[0], nf, 4, 2, 0), allow_norm=switch_norm),
                self.make_layer(nn.Conv2d(nf, nf * 2, 4, 2, 0, bias=self.norm is None)),
                self.make_layer(nn.Conv2d(nf * 2, nf * 4, 4, 2, 1, bias=self.norm is None)),
                self.make_layer(nn.Conv2d(nf * 4, nf * 8, 4, 2, 1, bias=self.norm is None)),
            ])
            self.linear = self.make_layer(nn.Linear(nf * 8 * 4 * 4, 512))
        elif cnn_kind == 'grouped': # custom grouped (6,950,912 parameters)
            nf = 32
            self.convs = nn.ModuleList([
                self.make_layer(nn.Conv2d(observation_space.shape[0], nf * 4, 4, 2, 0)),
                ChannelShuffle(nf * 4),
                self.make_layer(nn.Conv2d(nf * 4, nf * 8, 4, 2, 0, groups=8)),
                ChannelShuffle(nf * 8),
                self.make_layer(nn.Conv2d(nf * 8, nf * 16, 4, 2, 1, groups=16)),
                ChannelShuffle(nf * 16),
                self.make_layer(nn.Conv2d(nf * 16, nf * 32, 4, 2, 1, groups=32)),
                ChannelShuffle(nf * 32),
                self.make_layer(nn.Conv2d(nf * 32, nf * 8, 3, 1, 1, groups=8)),
            ])
            self.linear = self.make_layer(nn.Linear(nf * 8 * 4 * 4, 512))
        elif cnn_kind == 'depthwise':
            nf = 32
            self.convs = nn.ModuleList([
                # i x 84 x 84
                self.make_layer(nn.Conv2d(observation_space.shape[0], nf, 4, 2, 0)),
                # 1f x 40 x 40
                self.make_layer(nn.Conv2d(nf, nf, 4, 2, 0)),
                # (1 + 1)f x 20 x 20
                self.make_layer(nn.Conv2d(nf * 2, nf * 2, 4, 2, 1)),
                # (2 + 2)f x 10 x 10
                self.make_layer(nn.Conv2d(nf * 4, nf * 4, 4, 2, 1)),
                # (4 + 4)f x 5 x 5
            ])
            self.linear = self.make_layer(nn.Linear(nf * 8 * 4 * 4, 512))
        else:
            raise ValueError(cnn_kind)

        # create head
        self.head = head_factory(self.linear[0].out_features, self.pd)
        self.hidden_code_size = self.linear[0].in_features if cnn_hidden_code else self.linear[0].out_features

        self.reset_weights()

    def reset_weights(self):
        super().reset_weights()
        self.head.reset_weights()

    def make_layer(self, transf, allow_norm=True):
        parts = [transf]

        is_linear = isinstance(transf, nn.Linear)
        features = transf.out_features if is_linear else transf.out_channels
        norm_cls = None
        if self.norm is not None and allow_norm:
            if 'instance' in self.norm and not is_linear:
                norm_cls = nn.InstanceNorm2d(num_features=features, affine=True)
            if 'group' in self.norm:
                norm_cls = nn.GroupNorm(num_groups=features // 8, num_channels=features)
            if 'layer' in self.norm:
                norm_cls = nn.GroupNorm(num_groups=1, num_channels=features)
            if 'batch' in self.norm:
                norm_cls = nn.BatchNorm1d(features) if is_linear else nn.BatchNorm2d(features)
            if 'switch' in self.norm:
                norm_cls = SwitchableNorm1d(features) if is_linear else SwitchableNorm2d(features)
            if 'auto' in self.norm:
                norm_cls = AutoNorm(features)
            if norm_cls is not None:
                parts.append(norm_cls)

        parts.append(self.linear_activation() if is_linear else self.cnn_activation())

        return nn.Sequential(*parts)

    def _extract_features(self, input):
        x = input * 2 - 1
        for i, layer in enumerate(self.convs):
            # run conv layer
            if self.cnn_kind == 'depthwise' and i != 0:
                out = layer(x)
                pool = F.max_pool2d(x, 2)
                if pool.shape[2:] != out.shape[2:]:
                    pool = pool[:, :, :-1, :-1]
                x = torch.cat([pool, out], 1)
            else:
                x = layer(x)
            # log
            if self.do_log:
                self.log_conv_activations(i, x)
                # self.log_conv_filters(i, layer[0])
        return x

    def forward(self, input, hidden_code_input=False, only_hidden_code_output=False):
        log_policy_attention = self.do_log and input.is_leaf and not hidden_code_input and not only_hidden_code_output
        if not hidden_code_input:
            input = image_to_float(input)
            if log_policy_attention:
                input.requires_grad = True

            x = self._extract_features(input)

            x = x.view(x.size(0), -1)

            if not self.cnn_hidden_code:
                x = self.linear(x)

            # x.data.mul_(4).round_().div_(4)
        else:
            x = input

        hidden_code = x
        if only_hidden_code_output:
            return HeadOutput(hidden_code=hidden_code)

        if self.cnn_hidden_code:
            x = self.linear(x)

        ac_out = self.head(x)
        ac_out.hidden_code = hidden_code

        if not hidden_code_input:
            if self.do_log:
                self.logger.add_histogram('conv linear', x, self._step)

            if log_policy_attention:
                self.log_policy_attention(input, ac_out)

        return ac_out

    def parameters(self, hidden_code_l1_decay=0):
        if hidden_code_l1_decay == 0:
            return super().parameters()

        all_params = list(super().parameters())
        hidden_code_weight = self.linear[0].weight
        normal_params = [p for p in all_params if p is not hidden_code_weight]
        assert len(all_params) == len(set(normal_params)) + 1
        return [
            dict(params=normal_params),
            dict(params=[hidden_code_weight], l1_decay=hidden_code_l1_decay)
        ]


class Sega_CNNActor(CNNActor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nf = 32
        in_c = self.observation_space.shape[0]
        self.convs = nn.ModuleList([
            self.make_layer(nn.Conv2d(in_c,   nf,     8, 4, 0, bias=self.norm is None)),
            self.make_layer(nn.Conv2d(nf,     nf * 2, 6, 3, 0, bias=self.norm is None)),
            self.make_layer(nn.Conv2d(nf * 2, nf * 4, 4, 2, 0, bias=self.norm is None)),
        ])
        self.linear = self.make_layer(nn.Linear(1920, 512))
        self.reset_weights()