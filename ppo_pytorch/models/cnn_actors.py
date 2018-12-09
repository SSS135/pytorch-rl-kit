import math

import torch
import torch.nn as nn
from optfn.skip_connections import ResidualBlock
from torch import autograd
from torch.autograd import Variable

from .actors import FeatureExtractorBase, ModularActor
from .heads import StateValueQuantileHead, PolicyHead, StateValueHead
from .norm_factory import NormFactory
from .utils import make_conv_heatmap, image_to_float
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


class CNNFeatureExtractor(FeatureExtractorBase):
    """
    Convolution network.
    """
    def __init__(self, input_shape, cnn_kind='normal', cnn_activation=nn.ReLU, fc_activation=nn.ReLU, **kwargs):
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
        self.convs = None
        self.linear = None
        self._create_model()

    def _create_model(self):
        # create convolutional layers
        if self.cnn_kind == 'normal': # Nature DQN (1,683,456 parameters)
            self.convs = nn.ModuleList([
                self._make_cnn_layer(self.input_shape[0], 32, 8, 4, first_layer=True),
                self._make_cnn_layer(32, 64, 4, 2),
                self._make_cnn_layer(64, 64, 3, 1),
            ])
            self.linear = self._make_fc_layer(3136, 512)
        elif self.cnn_kind == 'large': # custom (2,066,432 parameters)
            nf = 32
            self.convs = nn.ModuleList([
                self._make_cnn_layer(self.input_shape[0], nf, 4, 2, 0, first_layer=True),
                self._make_cnn_layer(nf, nf * 2, 4, 2, 0),
                self._make_cnn_layer(nf * 2, nf * 4, 4, 2, 1),
                self._make_cnn_layer(nf * 4, nf * 8, 4, 2, 1),
            ])
            self.linear = self._make_fc_layer(nf * 8 * 4 * 4, 512)
        elif self.cnn_kind == 'grouped': # custom grouped (6,950,912 parameters)
            nf = 32
            self.convs = nn.ModuleList([
                self._make_cnn_layer(self.input_shape[0], nf * 4, 4, 2, 0, first_layer=True),
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
        elif self.cnn_kind == 'impala':
            def impala_block(c_in, c_out):
                return nn.Sequential(
                    nn.Conv2d(c_in, c_out, 3, 1, 1),
                    nn.MaxPool2d(3, 2),
                    ResidualBlock(
                        nn.ReLU(),
                        nn.Conv2d(c_out, c_out, 3, 1, 1),
                        nn.ReLU(),
                        nn.Conv2d(c_out, c_out, 3, 1, 1),
                    ),
                    ResidualBlock(
                        nn.ReLU(),
                        nn.Conv2d(c_out, c_out, 3, 1, 1),
                        nn.ReLU(),
                        nn.Conv2d(c_out, c_out, 3, 1, 1),
                    )
                )
            self.convs = nn.Sequential(
                impala_block(4, 16),
                impala_block(16, 32),
                impala_block(32, 32),
                nn.ReLU(),
            )
            self.linear = nn.Sequential(
                nn.Linear(2592, 256),
                nn.ReLU(),
            )
        else:
            raise ValueError(self.cnn_kind)

    @property
    def output_size(self):
        return self.linear[0].out_features

    def _make_fc_layer(self, in_features, out_features, first_layer=False):
        bias = self.norm_factory is None or not self.norm_factory.disable_bias or not self.norm_factory.allow_fc
        return self._make_layer(nn.Linear(in_features, out_features, bias=bias), first_layer=first_layer)

    def _make_cnn_layer(self, *args, first_layer=False, **kwargs):
        bias = self.norm_factory is None or not self.norm_factory.disable_bias or not self.norm_factory.allow_cnn
        return self._make_layer(nn.Conv2d(*args, **kwargs, bias=bias), first_layer=first_layer)

    def _make_layer(self, transf, first_layer=False):
        is_linear = isinstance(transf, nn.Linear)
        features = transf.out_features if is_linear else transf.out_channels

        parts = [transf]
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

    def forward(self, input, logger=None, cur_step=None, **kwargs):
        input = image_to_float(input)

        x = self._extract_features(input)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        if logger is not None:
            logger.add_histogram('conv linear', x, cur_step)

        return x

    def _log_conv_activations(self, index: int, x: torch.Tensor, logger, cur_step):
        with torch.no_grad():
            img = x[0].unsqueeze(1).clone()
            img = make_conv_heatmap(img)
            img = make_grid(img, nrow=round(math.sqrt(x.shape[1])), normalize=False, fill_value=0.1)
            logger.add_image('conv activations {} img'.format(index), img, cur_step)
            logger.add_histogram('conv activations {} hist'.format(index), x[0], cur_step)

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
            logger.add_image('conv featrues {} img'.format(index), img, cur_step)
            logger.add_histogram('conv features {} hist'.format(index), conv.weight, cur_step)

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
            logger.add_image('state attention', img, cur_step)


class Sega_CNNFeatureExtractor(CNNFeatureExtractor):
    def _create_model(self):
        nf = 32
        in_c = self.input_shape[0]
        self.convs = nn.ModuleList([
            self._make_layer(nn.Conv2d(in_c, nf, 8, 4, 0, bias=self.norm_factory is None)),
            self._make_layer(nn.Conv2d(nf, nf * 2, 6, 3, 0, bias=self.norm_factory is None)),
            self._make_layer(nn.Conv2d(nf * 2, nf * 4, 4, 2, 0, bias=self.norm_factory is None)),
        ])
        self.linear = self._make_layer(nn.Linear(1920, 512))


def create_ppo_cnn_actor(observation_space, action_space, cnn_kind='normal',
                         cnn_activation=nn.ReLU, fc_activation=nn.ReLU, norm_factory: NormFactory=None,
                         iqn=False, split_policy_value_network=False):
    assert len(observation_space.shape) == 3
    pd = make_pd(action_space)

    create_fx = lambda: CNNFeatureExtractor(observation_space.shape, cnn_kind,
                                            cnn_activation, fc_activation, norm_factory=norm_factory)

    if split_policy_value_network:
        fx_policy, fx_value = create_fx(), create_fx()
    else:
        fx_policy = fx_value = create_fx()

    value_head = (StateValueQuantileHead if iqn else StateValueHead)(fx_value.output_size)
    policy_head = PolicyHead(fx_policy.output_size, pd)

    if split_policy_value_network:
        models = {fx_policy: dict(logits=policy_head), fx_value: dict(state_values=value_head)}
    else:
        models = {fx_policy: dict(logits=policy_head, state_values=value_head)}
    return ModularActor(models)