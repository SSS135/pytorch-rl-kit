import math

import torch
import torch.nn as nn
from torch import autograd
from torch.autograd import Variable

from .actors import Actor
from .heads import HeadOutput
from .utils import make_conv_heatmap, image_to_float
from ..common.make_grid import make_grid


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


class CNNActor(Actor):
    """
    Convolution network.
    """
    def __init__(self, observation_space, action_space, head_factory, cnn_kind='normal', *args,
                 cnn_activation=nn.ReLU, linear_activation=nn.ReLU, cnn_hidden_code=False, hidden_code_type='input', **kwargs):
        """
        Args:
            observation_space: Env's observation space
            action_space: Env's action space
            head_factory: Function which accept (hidden vector size, `ProbabilityDistribution`) and return `HeadBase`
            cnn_kind: Type of cnn.
                'normal' - CNN from Nature DQN paper (Mnih et al. 2015)
                'custom' - largest CNN of custom structure
            cnn_activation: Activation function
        """
        super().__init__(observation_space, action_space, head_factory, *args, **kwargs)
        self.cnn_activation = cnn_activation
        self.linear_activation = linear_activation
        self.cnn_kind = cnn_kind
        self.cnn_hidden_code = cnn_hidden_code

        # create convolutional layers
        if cnn_kind == 'normal': # Nature DQN (1,683,456 parameters)
            self.convs = nn.ModuleList([
                self._make_cnn_layer(observation_space.shape[0], 32, 8, 4, first_layer=True),
                self._make_cnn_layer(32, 64, 4, 2),
                self._make_cnn_layer(64, 64, 3, 1),
            ])
            self.linear = self._make_fc_layer(3136, 512)
        elif cnn_kind == 'large': # custom (2,066,432 parameters)
            nf = 32
            self.convs = nn.ModuleList([
                self._make_cnn_layer(observation_space.shape[0], nf, 4, 2, 0, first_layer=True),
                self._make_cnn_layer(nf, nf * 2, 4, 2, 0),
                self._make_cnn_layer(nf * 2, nf * 4, 4, 2, 1),
                self._make_cnn_layer(nf * 4, nf * 8, 4, 2, 1),
            ])
            self.linear = self._make_fc_layer(nf * 8 * 4 * 4, 512)
        elif cnn_kind == 'grouped': # custom grouped (6,950,912 parameters)
            nf = 32
            self.convs = nn.ModuleList([
                self._make_cnn_layer(observation_space.shape[0], nf * 4, 4, 2, 0, first_layer=True),
                ChannelShuffle(nf * 4),
                self._make_cnn_layer(nf * 4, nf * 8, 4, 2, 0, groups=8),
                ChannelShuffle(nf * 8),
                self._make_cnn_layer(nf * 8, nf * 16, 4, 2, 1, groups=16),
                ChannelShuffle(nf * 16),
                self._make_cnn_layer(nf * 16, nf * 32, 4, 2, 1, groups=32),
                ChannelShuffle(nf * 32),
                self._make_cnn_layer(nf * 32, nf * 8, 3, 1, 1, groups=8),
            ])
            self.linear = self._make_fc_layer(nf * 8 * 4 * 4, 512)
        else:
            raise ValueError(cnn_kind)

        # create head
        self.hidden_code_size = self.linear[0].in_features if cnn_hidden_code else self.linear[0].out_features
        self._init_heads(self.hidden_code_size)

        self.reset_weights()

    def _make_fc_layer(self, in_features, out_features, first_layer=False):
        bias = self.norm is None or not self.norm.disable_bias
        return self._make_layer(nn.Linear(in_features, out_features, bias=bias), first_layer=first_layer)

    def _make_cnn_layer(self, *args, first_layer=False, **kwargs):
        bias = self.norm is None or not self.norm.disable_bias
        return self._make_layer(nn.Conv2d(*args, **kwargs, bias=bias), first_layer=first_layer)

    def _make_layer(self, transf, first_layer=False):
        is_linear = isinstance(transf, nn.Linear)
        features = transf.out_features if is_linear else transf.out_channels

        parts = [transf]
        if self.norm is not None and \
                (self.norm.allow_after_first_layer or not first_layer) and \
                (self.norm.allow_fc if is_linear else self.norm.allow_cnn):
            func = self.norm.create_fc_norm if is_linear else self.norm.create_cnn_norm
            parts.append(func(features, first_layer))
        parts.append(self.linear_activation() if is_linear else self.cnn_activation())
        return nn.Sequential(*parts)

    def _extract_features(self, x):
        #x = input - input.median() # * 2 - 1
        for i, layer in enumerate(self.convs):
            # run conv layer
            x = layer(x)
            # log
            if self.do_log:
                self._log_conv_activations(i, x)
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
        else:
            x = input

        hidden_code = x
        if only_hidden_code_output:
            return HeadOutput(hidden_code=hidden_code)

        if self.cnn_hidden_code:
            x = self.linear(x)

        ac_out = self._run_heads(x)
        ac_out.hidden_code = hidden_code

        if not hidden_code_input:
            if self.do_log:
                self.logger.add_histogram('conv linear', x, self._step)

            if log_policy_attention:
                self._log_policy_attention(input, ac_out)

        return ac_out

    def _log_conv_activations(self, index: int, x: Variable):
        with torch.no_grad():
            img = x[0].unsqueeze(1).clone()
            img = make_conv_heatmap(img)
            img = make_grid(img, nrow=round(math.sqrt(x.shape[1])), normalize=False, fill_value=0.1)
            self.logger.add_image('conv activations {} img'.format(index), img, self._step)
            self.logger.add_histogram('conv activations {} hist'.format(index), x[0], self._step)

    def _log_conv_filters(self, index: int, conv: nn.Conv2d):
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

    def _log_policy_attention(self, states, head_out):
        states_grad = autograd.grad(
            head_out.probs.abs().mean() + head_out.state_value.abs().mean(), states,
            only_inputs=True, retain_graph=True)[0]
        with torch.no_grad():
            img = states_grad[:4]
            img.abs_()
            img /= img.view(4, -1).pow(2).mean(1).sqrt_().add_(1e-5).view(4, 1, 1, 1)
            img = img.view(-1, 1, *img.shape[2:]).abs()
            img = make_grid(img, 4, normalize=True, fill_value=0.1)
            self.logger.add_image('state attention', img, self._step)


class Sega_CNNActor(CNNActor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, cnn_kind='large', **kwargs)
        nf = 32
        in_c = self.observation_space.shape[0]
        self.convs = nn.ModuleList([
            self._make_layer(nn.Conv2d(in_c, nf, 8, 4, 0, bias=self.norm is None)),
            self._make_layer(nn.Conv2d(nf, nf * 2, 6, 3, 0, bias=self.norm is None)),
            self._make_layer(nn.Conv2d(nf * 2, nf * 4, 4, 2, 0, bias=self.norm is None)),
        ])
        self.linear = self._make_layer(nn.Linear(1920, 512))
        self.reset_weights()