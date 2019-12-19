import math
from functools import partial

import torch
import torch.nn as nn
from optfn.skip_connections import ResidualBlock
from torch import autograd
from torch.autograd import Variable

from .actors import FeatureExtractorBase, create_ppo_actor
from .norm_factory import NormFactory
from .utils import make_conv_heatmap, image_to_float
from ..common.make_grid import make_grid
from ..config import Linear
from .utils import fixup_init
from .activation_norm import ActivationNorm, ActivationNormWrapper


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


class CNNFeatureExtractor(FeatureExtractorBase):
    """
    Convolution network.
    """
    def __init__(self, input_shape, cnn_kind='normal',
                 cnn_activation=partial(nn.ReLU, inplace=True), fc_activation=partial(nn.ReLU, inplace=True),
                 add_positional_features=False, normalize_input=False, **kwargs):
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
        self.linear_activation = fc_activation
        self.cnn_kind = cnn_kind
        self.add_positional_features = add_positional_features
        self.normalize_input = normalize_input
        self.convs = None
        self.linear = None
        self._prev_positions = None
        self._create_model()

    def _create_model(self):
        input_channels = self.input_shape[0] + (2 if self.add_positional_features else 0)
        # create convolutional layers
        if self.cnn_kind == 'normal': # Nature DQN (1,683,456 parameters)
            self.convs = nn.ModuleList([
                self._make_cnn_layer(input_channels, 32, 8, 4, first_layer=True),
                self._make_cnn_layer(32, 64, 4, 2),
                self._make_cnn_layer(64, 64, 3, 1),
            ])
            self.linear = self._make_fc_layer(self._calc_linear_size(), 512)
        elif self.cnn_kind == 'large': # custom (2,066,432 parameters)
            nf = 32
            self.convs = nn.ModuleList([
                self._make_cnn_layer(input_channels, nf, 4, 2, 0, first_layer=True),
                self._make_cnn_layer(nf, nf * 2, 4, 2, 0),
                self._make_cnn_layer(nf * 2, nf * 4, 4, 2, 1),
                self._make_cnn_layer(nf * 4, nf * 8, 4, 2, 1),
            ])
            self.linear = self._make_fc_layer(self._calc_linear_size(), 512)
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
            self.linear = self._make_fc_layer(self._calc_linear_size(), 512)
        elif self.cnn_kind == 'impala':
            c_mult = 2
            def cnn_norm_fn(num_c):
                return (self.norm_factory.create_cnn_norm(num_c, False),) \
                    if self.norm_factory is not None and self.norm_factory.allow_cnn else ()
            def fc_norm_fn(num_c):
                return (self.norm_factory.create_fc_norm(num_c, False),) \
                    if self.norm_factory is not None and self.norm_factory.allow_fc else ()
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
                    # ActivationNorm(c_out, c_out // 16),
                )
            self.convs = nn.Sequential(
                impala_block(input_channels, 16 * c_mult),
                impala_block(16 * c_mult, 32 * c_mult),
                impala_block(32 * c_mult, 32 * c_mult),
                # impala_block(64 * c_mult, 64 * c_mult),
                nn.Sequential(
                    *cnn_norm_fn(32 * c_mult),
                    nn.ReLU(True),
                )
            )
            self.linear = nn.Sequential(
                Linear(self._calc_linear_size(), 256),
                # ActivationNorm(256, 256 // 16),
                *fc_norm_fn(256),
                nn.ReLU(True),
            )
        else:
            raise ValueError(self.cnn_kind)

    @property
    def output_size(self):
        return self.linear[0].out_features

    # def reset_weights(self):
    #     super().reset_weights()
    #     if self.cnn_kind == 'impala':
    #         fixup_init(self.convs)

    def _calc_linear_size(self):
        shape = 1, self.input_shape[0] + (2 if self.add_positional_features else 0), *self.input_shape[1:]
        out_shape = self._extract_features(torch.randn(shape)).shape
        return out_shape[1] * out_shape[2] * out_shape[3]

    def _make_fc_layer(self, in_features, out_features, first_layer=False, activation_norm=True):
        bias = self.norm_factory is None or not self.norm_factory.disable_bias or not self.norm_factory.allow_fc
        return self._make_layer(Linear(in_features, out_features, bias=bias),
                                first_layer=first_layer, activation_norm=activation_norm)

    def _make_cnn_layer(self, *args, first_layer=False, activation_norm=True, **kwargs):
        bias = self.norm_factory is None or not self.norm_factory.disable_bias or not self.norm_factory.allow_cnn
        return self._make_layer(nn.Conv2d(*args, **kwargs, bias=bias),
                                first_layer=first_layer, activation_norm=activation_norm)

    def _make_layer(self, transf, first_layer=False, activation_norm=True):
        is_linear = isinstance(transf, nn.Linear) or isinstance(transf, Linear)
        features = transf.out_features if is_linear else transf.out_channels

        # parts = [ActivationNormWrapper(transf)]
        parts = [transf]
        if activation_norm:
            parts.append(ActivationNorm())
        if self.norm_factory is not None and \
                (self.norm_factory.allow_after_first_layer or not first_layer) and \
                (self.norm_factory.allow_fc if is_linear else self.norm_factory.allow_cnn):
            func = self.norm_factory.create_fc_norm if is_linear else self.norm_factory.create_cnn_norm
            parts.append(func(features, first_layer))
        parts.append(self.linear_activation() if is_linear else self.cnn_activation())
        return nn.Sequential(*parts)

    def _extract_features(self, x, logger=None, cur_step=None):
        for i, layer in enumerate(self.convs):
            x = layer(x)
            if logger is not None:
                self._log_conv_activations(i, x, logger, cur_step)
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
        input = image_to_float(input)
        if self.normalize_input:
            input = input * 2 - 1
        if self.add_positional_features:
            input = self._add_position_features(input)

        x = self._extract_features(input, logger, cur_step)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        if logger is not None:
            logger.add_histogram('conv_activations_linear', x, cur_step)

        return x

    def _log_conv_activations(self, index: int, x: torch.Tensor, logger, cur_step):
        with torch.no_grad():
            img = x[0].unsqueeze(1).clone()
            img = make_conv_heatmap(img)
            img = make_grid(img, nrow=round(math.sqrt(x.shape[1])), normalize=False, fill_value=0.1)
            logger.add_image('conv_activations_{}_img'.format(index), img, cur_step)
            logger.add_histogram('conv_activations_{}_hist'.format(index), x[0], cur_step)

    def _log_conv_filters(self, index: int, conv: nn.Conv2d, logger, cur_step):
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

    def _log_policy_attention(self, states, head_out, logger, cur_step):
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


class Sega_CNNFeatureExtractor(CNNFeatureExtractor):
    def _create_model(self):
        nf = 32
        in_c = self.input_shape[0]
        self.convs = nn.ModuleList([
            self._make_layer(nn.Conv2d(in_c, nf, 8, 4, 0, bias=self.norm_factory is None)),
            self._make_layer(nn.Conv2d(nf, nf * 2, 6, 3, 0, bias=self.norm_factory is None)),
            self._make_layer(nn.Conv2d(nf * 2, nf * 4, 4, 2, 0, bias=self.norm_factory is None)),
        ])
        self.linear = self._make_layer(Linear(1920, 512))


def create_ppo_cnn_actor(observation_space, action_space, cnn_kind='normal',
                         cnn_activation=nn.ReLU, fc_activation=nn.ReLU, norm_factory: NormFactory=None,
                         split_policy_value_network=False, num_out=1,
                         add_positional_features=False, normalize_input=False):
    assert len(observation_space.shape) == 3

    def fx_factory(): return CNNFeatureExtractor(
        observation_space.shape, cnn_kind, cnn_activation, fc_activation, norm_factory=norm_factory,
        add_positional_features=add_positional_features, normalize_input=normalize_input)
    return create_ppo_actor(action_space, fx_factory, split_policy_value_network, num_out=num_out)