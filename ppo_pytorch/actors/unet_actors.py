import math
from functools import partial

import torch
import torch.nn as nn
from optfn.skip_connections import ResidualBlock
from ..common.multi_cnn_space import MultiCNNSpace

from .cnn_actors import log_conv_activations
from .fc_actors import FCFeatureExtractor
from .rnn_actors import RNNFeatureExtractor
from ..common.probability_distributions import make_pd
from torch import autograd
from torch.autograd import Variable
from torch import Tensor

from .actors import FeatureExtractorBase, create_ppo_actor, create_impala_actor
from .norm_factory import NormFactory
from .utils import make_conv_heatmap, image_to_float
from ..common.make_grid import make_grid
from ..common.activation_norm import ActivationNorm
import torch.nn.functional as F

N = 8


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            #             nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            #             nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64 // N)
        self.down1 = Down(64 // N, 128 // N)
        self.down2 = Down(128 // N, 256 // N)
        self.down3 = Down(256 // N, 512 // N)
        factor = 2 if bilinear else 1
        self.down4 = Down(512 // N, 1024 // N // factor)
        self.up1 = Up(1024 // N, 512 // N // factor, bilinear)
        self.up2 = Up(512 // N, 256 // N // factor, bilinear)
        self.up3 = Up(256 // N, 128 // N // factor, bilinear)
        self.up4 = Up(128 // N, 64 // N, bilinear)

    def forward(self, x, logger, cur_step):
        def log(i, x):
            if logger is not None:
                log_conv_activations(i, x, logger, cur_step)

        x1 = self.inc(x)
        log(0, x1)
        x2 = self.down1(x1)
        log(1, x2)
        x3 = self.down2(x2)
        log(2, x3)
        x4 = self.down3(x3)
        log(3, x4)
        x5 = self.down4(x4)
        log(4, x5)
        x = self.up1(x5, x4)
        log(5, x)
        x = self.up2(x, x3)
        log(6, x)
        x = self.up3(x, x2)
        log(7, x)
        x = self.up4(x, x1)
        log(8, x)
        return x


class UNetFeatureExtractor(FeatureExtractorBase):
    """
    Convolution network.
    """
    def __init__(self, input_shape, cnn_activation=partial(nn.ReLU, inplace=True),
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
        self.add_positional_features = add_positional_features
        self.normalize_input = normalize_input
        self.activation_norm = activation_norm
        self.model = None
        self._prev_positions = None
        self._create_model()
        self._output_size = 64 // N

    def _create_model(self):
        input_channels = self.input_shape[0] + (2 if self.add_positional_features else 0)
        self.model = UNet(input_channels)

    @property
    def output_size(self):
        return self._output_size

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

        x = self.model(input, logger, cur_step)

        if logger is not None:
            logger.add_histogram('conv_activations_linear', x, cur_step)

        out_shape = (*input_shape[:-3], self.output_size, input_shape[-2] * input_shape[-1])
        return x.view(out_shape).transpose(-1, -2)


def create_impala_unet_actor(observation_space, action_space, activation=partial(nn.ReLU, inplace=True), num_values=1,
                            add_positional_features=False, normalize_input=False, goal_size=0):
    assert len(observation_space.shape) == 3
    assert isinstance(action_space, MultiCNNSpace)

    def fx_factory():
        return UNetFeatureExtractor(
            observation_space.shape, activation,
            add_positional_features=add_positional_features, normalize_input=normalize_input)
    return create_impala_actor(action_space, fx_factory, split_policy_value_network=False, num_values=num_values,
                               is_recurrent=False)