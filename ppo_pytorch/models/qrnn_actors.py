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
from ..common.qrnn import QRNN, DenseQRNN


class QRNNActor(Actor):
    def __init__(self, obs_space: gym.Space, action_space: gym.Space, head_factory: Callable,
                 hidden_size=128, num_layers=3, **kwargs):
        """
        Args:
            obs_space: Env's observation space
            action_space: Env's action space
            head_factory: Function which accept (hidden vector size, `ProbabilityDistribution`) and return `HeadBase`
            hidden_sizes: List of hidden layers sizes
            activation: Activation function
        """
        super().__init__(obs_space, action_space, **kwargs)
        obs_len = int(np.product(obs_space.shape))
        self.qrnn = DenseQRNN(obs_len, hidden_size, num_layers)
        self.head = head_factory(hidden_size, self.pd)
        self.reset_weights()

    def reset_weights(self):
        super().reset_weights()
        self.head.reset_weights()

    def forward(self, input, memory, done_flags):
        x, next_memory = self.qrnn(input, memory, done_flags)
        head = self.head(x)
        return head, next_memory