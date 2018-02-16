import math
from functools import partial
from typing import Optional, List, Callable

import gym
import numpy as np
import torch.nn as nn
import torch.nn.init as init
from torch import autograd
from torch.autograd import Variable

from .utils import weights_init, make_conv_heatmap, image_to_float
from ..common.layer_norm import LayerNorm1d, LayerNorm2d
from ..common.make_grid import make_grid
from ..common.probability_distributions import make_pd
from .actors import Actor


class QRNNActor(Actor):
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