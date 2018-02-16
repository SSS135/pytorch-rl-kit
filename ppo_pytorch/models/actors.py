import math
from functools import partial
from typing import Optional, List, Callable

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from ..common.pre_act_res_net import PreActBlockCustom, Empty
from torch import autograd
from torch.autograd import Variable

from .utils import weights_init, make_conv_heatmap, image_to_float
from ..common.layer_norm import LayerNorm1d, LayerNorm2d
from ..common.make_grid import make_grid
from ..common.probability_distributions import make_pd


class ActorOutput:
    """
    Output of `Actor`. Different actors may fill different arguments.
    """

    def __init__(self, probs=None, state_values=None, conv_out=None, hidden_code=None,
                 action_values=None, head_raw=None):
        self.probs = probs
        self.state_values = state_values
        self.conv_out = conv_out
        self.hidden_code = hidden_code
        self.action_values = action_values
        self.head_raw = head_raw


class Actor(nn.Module):
    """
    Base class for network in reinforcement learning algorithms.
    """

    def __init__(self, obs_space: gym.Space, action_space: gym.Space, norm: str = None,
                 weight_init=init.orthogonal, weight_init_gain=math.sqrt(2)):
        super().__init__()
        self.obs_space = obs_space
        self.action_space = action_space
        self.weight_init_gain = weight_init_gain
        self.weight_init = weight_init
        self.norm = norm
        assert norm in (None, 'layer', 'batch')

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

    @staticmethod
    def create_mlp(in_size: int, out_size: Optional[int], hidden_sizes: List[int],
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
                'weight' - weight normalization

        Returns: `nn.Sequential` of layers. Each layer is also `nn.Sequential` containing (linear, [norm], activation).
            If `out_size` is not None, last layer is just linear transformation, without norm or activation.

        """
        assert norm in (None, 'layer', 'batch', 'weight')
        seq = []
        for i in range(len(hidden_sizes)):
            n_in = in_size if i == 0 else hidden_sizes[i - 1]
            n_out = hidden_sizes[i]
            layer = []
            layer.append(nn.Linear(n_in, n_out))
            if norm == 'layer':
                layer.append(LayerNorm1d(n_out))
            elif norm == 'batch':
                layer.append(nn.BatchNorm1d(n_out))
            layer.append(activation())
            seq.append(nn.Sequential(*layer))
        if out_size is not None:
            seq.append(nn.Linear(hidden_sizes[-1], out_size))
        seq = nn.Sequential(*seq)
        return seq


class MLPActor(Actor):
    """
    Fully connected network.
    """

    def __init__(self, obs_space: gym.Space, action_space: gym.Space, head_factory: Callable,
                 hidden_sizes=(128, 128), activation=nn.ELU, **kwargs):
        """
        Args:
            obs_space: Env's observation space
            action_space: Env's action space
            head_factory: Function which accept (hidden vector size, `ProbabilityDistribution`) and return `HeadBase`
            hidden_sizes: List of hidden layers sizes
            activation: Activation function
        """
        super().__init__(obs_space, action_space, **kwargs)
        self.hidden_sizes = hidden_sizes
        self.activation = activation

        obs_len = int(np.product(obs_space.shape))

        self.linear = self.create_mlp(obs_len, None, hidden_sizes, activation, self.norm)
        self.head = head_factory(hidden_sizes[-1], self.pd)
        self.reset_weights()

    def reset_weights(self):
        super().reset_weights()
        self.head.reset_weights()

    def forward(self, input):
        x = input
        for i, layer in enumerate(self.linear):
            x = layer(x)
            if self.do_log:
                self.logger.add_histogram(f'layer {i} output', x, self._step)
        hidden = x
        head = self.head(x)
        head.hidden_code = hidden
        return head


class CNNActor(Actor):
    """
    Convolution network.
    """
    def __init__(self, obs_space, action_space, head_factory, cnn_kind='large',
                 cnn_activation=nn.ReLU, linear_activation=nn.ReLU, dropout=0, **kwargs):        """
        Args:
            obs_space: Env's observation space
            action_space: Env's action space
            head_factory: Function which accept (hidden vector size, `ProbabilityDistribution`) and return `HeadBase`
            cnn_kind: Type of cnn.
                'small' - small CNN from arxiv DQN paper (Mnih et al. 2013)
                'large' - bigger CNN from Nature DQN paper (Mnih et al. 2015)
                'custom' - largest CNN of custom structure
            cnn_activation: Activation function
        """
        super().__init__(obs_space, action_space, **kwargs)
        self.cnn_activation = cnn_activation
        self.linear_activation = linear_activation
        self.cnn_kind = cnn_kind
        # assert cnn_kind in ('small', 'large', 'custom')

        def make_layer(transf):
            parts = [transf]

            is_linear = isinstance(transf, nn.Linear)
            features = transf.out_features if is_linear else transf.out_channels
            if self.norm == 'layer':
                norm_cls = LayerNorm1d if is_linear else partial(nn.InstanceNorm2d, affine=True)
            elif self.norm == 'batch':
                norm_cls = nn.BatchNorm1d if is_linear else nn.BatchNorm2d
            else:
                norm_cls = None
            if norm_cls is not None and not is_linear:
                parts.append(norm_cls(features))

            parts.append(linear_activation() if is_linear else cnn_activation())

            return nn.Sequential(*parts)

        # create convolutional layers
        if cnn_kind == 'small': # from A3C paper (675,840 parameters)
            self.convs = nn.ModuleList([
                make_layer(nn.Conv2d(obs_space.shape[0], 16, 8, 4)),
                make_layer(nn.Conv2d(16, 32, 4, 2)),
            ])
            self.linear = make_layer(nn.Linear(2592, 256))
        elif cnn_kind == 'large': # Nature DQN (1,683,456 parameters)
            self.convs = nn.ModuleList([
                make_layer(nn.Conv2d(obs_space.shape[0], 32, 8, 4)),
                make_layer(nn.Conv2d(32, 64, 4, 2)),
                make_layer(nn.Conv2d(64, 64, 3, 1)),
            ])
            self.linear = make_layer(nn.Linear(3136, 512))
        elif cnn_kind == 'custom': # custom (6,950,912 parameters)
            nf = 64
            self.convs = nn.ModuleList([
                make_layer(nn.Conv2d(obs_space.shape[0], nf, 4, 2, 1)),
                make_layer(nn.Conv2d(nf, nf * 2, 4, 2, 0)),
                make_layer(nn.Conv2d(nf * 2, nf * 4, 4, 2, 1)),
                make_layer(nn.Conv2d(nf * 4, nf * 8, 4, 2, 0)),
                nn.Dropout2d(dropout),
            ])
            self.linear = make_layer(nn.Linear(nf * 8 * 4 * 4, 512))
        elif cnn_kind == 'rl_a3c_pytorch': # 622,720 parameters
            # https://github.com/dgriff777/rl_a3c_pytorch/blob/master/model.py
            nf = 32
            self.convs = nn.ModuleList([
                make_layer(nn.Conv2d(obs_space.shape[0], nf, 5, 1, 2)),
                nn.MaxPool2d(2, 2),
                make_layer(nn.Conv2d(nf, nf, 5, 1, 1)),
                nn.MaxPool2d(2, 2),
                make_layer(nn.Conv2d(nf, 2 * nf, 4, 1, 1)),
                nn.MaxPool2d(2, 2),
                make_layer(nn.Conv2d(2 * nf, 2 * nf, 3, 1, 1)),
                nn.MaxPool2d(2, 2),
            ])
            self.linear = make_layer(nn.Linear(1024, 512))
        elif cnn_kind == 'depthwise':
            nf = 64
            self.convs = nn.ModuleList([
                # i x 84 x 84
                make_layer(nn.Conv2d(obs_space.shape[0], nf, 4, 2, 0)),
                # 1f x 40 x 40
                make_layer(nn.Conv2d(nf, nf, 4, 2, 1)),
                # (1 + 1)f x 20 x 20
                make_layer(nn.Conv2d(nf * 2, nf * 2, 4, 2, 1)),
                # (2 + 2)f x 10 x 10
                make_layer(nn.Conv2d(nf * 4, nf * 4, 4, 2, 1)),
                # (4 + 4)f x 5 x 5
            ])
            self.linear = make_layer(nn.Linear(nf * 8 * 5 * 5, 512))
        elif cnn_kind == 'resnet':
            nf = 32
            if self.norm == 'layer':
                norm = partial(nn.InstanceNorm2d, affine=True)
            elif self.norm == 'batch':
                norm = nn.BatchNorm2d
            else:
                norm = None

            self.convs = nn.ModuleList([
                # i x 84 x 84
                nn.Conv2d(obs_space.shape[0], nf, 4, 2, 0),
                nn.Sequential(
                    PreActBlockCustom(nf, nf, 1, activation=cnn_activation, norm=norm),
                    PreActBlockCustom(nf, nf, 1, activation=cnn_activation, norm=norm),
                ),
                nn.Sequential(
                    PreActBlockCustom(nf, 2 * nf, 2, activation=cnn_activation, norm=norm),
                    PreActBlockCustom(2 * nf, 2 * nf, 1, activation=cnn_activation, norm=norm),
                ),
                nn.Sequential(
                    PreActBlockCustom(2 * nf, 4 * nf, 2, activation=cnn_activation, norm=norm),
                    PreActBlockCustom(4 * nf, 4 * nf, 1, activation=cnn_activation, norm=norm),
                ),
                nn.Sequential(
                    PreActBlockCustom(4 * nf, 8 * nf, 2, activation=cnn_activation, norm=norm),
                    PreActBlockCustom(8 * nf, 8 * nf, 1, activation=cnn_activation, norm=norm),
                ),
                nn.Conv2d(8 * nf, nf, 1, 1, bias=False),
                # nn.Sequential(
                #     norm(8 * nf),
                #     nn.Conv2d(8 * nf, nf, 1, 1, bias=False),
                #     cnn_activation(),
                # ),
            ])
            self.linear = make_layer(nn.Linear(nf * 6 * 6, 512))
        else:
            raise NotImplementedError(cnn_kind)

        # create head
        self.head = head_factory(self.linear[0].out_features, self.pd)

        self.reset_weights()

    def reset_weights(self):
        super().reset_weights()
        self.head.reset_weights()

    def forward(self, input) -> ActorOutput:
        log_policy_attention = self.do_log and input.is_leaf
        input = image_to_float(input)
        if log_policy_attention:
            input = Variable(input.data, requires_grad=True)

        x = input
        for i, layer in enumerate(self.convs):
            # run conv layer
            if self.cnn_kind == 'depthwise' and i != 0:
                out = layer(x)
                x = torch.cat([F.max_pool2d(x, 2), out], 1)
            else:
                x = layer(x)
            # log
            if self.do_log and isinstance(layer, nn.Sequential) and isinstance(layer[0], nn.Conv2d):
                self.log_conv_activations(i, layer[0], x)
                self.log_conv_filters(i, layer[0])

        # flatten convolution output
        x = x.view(x.size(0), -1)
        # run linear layer
        x = self.linear(x)

        ac_out = self.head(x)

        if self.do_log:
            self.logger.add_histogram('conv linear', x, self._step)

        if log_policy_attention:
            self.log_policy_attention(input, ac_out)

        return ac_out

    def log_conv_activations(self, index: int, conv: nn.Conv2d, x: Variable):
        img = x[0].data.unsqueeze(1).clone()
        img = make_conv_heatmap(img)
        img = make_grid(img, nrow=round(math.sqrt(conv.out_channels)), normalize=False, fill_value=0.1)
        self.logger.add_image('conv activations {} img'.format(index), img, self._step)
        self.logger.add_histogram('conv activations {} hist'.format(index), x[0], self._step)

    def log_conv_filters(self, index: int, conv: nn.Conv2d):
        channels = conv.in_channels * conv.out_channels
        shape = conv.weight.data.shape
        kernel_h, kernel_w = shape[2], shape[3]
        img = conv.weight.data.view(channels, 1, kernel_h, kernel_w).clone()
        max_img_size = 100 * 5
        img_size = channels * math.sqrt(kernel_h * kernel_w)
        if img_size > max_img_size:
            channels = channels * (max_img_size / img_size)
            channels = math.ceil(math.sqrt(channels)) ** 2
            img = img[:channels]
        img = make_conv_heatmap(img, scale=2 * img.std())
        img = make_grid(img, nrow=round(math.sqrt(channels)), normalize=False, fill_value=0.1)
        self.logger.add_image('conv featrues {} img'.format(index), img, self._step)
        self.logger.add_histogram('conv features {} hist'.format(index), conv.weight.data, self._step)

    def log_policy_attention(self, states, head_out):
        states_grad = autograd.grad(
            head_out.probs.abs().mean() + head_out.state_values.abs().mean(), states,
            only_inputs=True, retain_graph=True)[0]
        img = states_grad.data[:4]
        img.abs_()
        img /= img.view(4, -1).pow(2).mean(1).sqrt_().add_(1e-5).view(4, 1, 1, 1)
        img = img.view(-1, 1, *img.shape[2:]).abs()
        # img = make_conv_heatmap(img, scale=2*img.std())
        img = make_grid(img, 4, normalize=True, fill_value=0.1)
        self.logger.add_image('state attention', img, self._step)