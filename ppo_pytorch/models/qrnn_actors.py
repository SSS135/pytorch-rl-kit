import math
from functools import partial
from typing import Optional, List, Callable

import gym
import numpy as np
import torch.nn as nn
import torch.nn.init as init
from torch import autograd
from torch.autograd import Variable
import torch
import torch.nn.functional as F

from .utils import weights_init, make_conv_heatmap, image_to_float
from optfn.layer_norm import LayerNorm1d, LayerNorm2d
from ..common.make_grid import make_grid
from ..common.probability_distributions import make_pd
from .actors import Actor, CNNActor, ActorOutput
from optfn.qrnn import QRNN, DenseQRNN


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


class CNN_QRNNActor(CNNActor):
    def __init__(self, obs_space, action_space, head_factory, cnn_kind='large',
                 cnn_activation=nn.ReLU, linear_activation=nn.ReLU,
                 qrnn_hidden_size=512, qrnn_layers=3, dropout=0, **kwargs):
        """
        Args:
            obs_space: Env's observation space
            action_space: Env's action space
            head_factory: Function which accept (hidden vector size, `ProbabilityDistribution`) and return `HeadBase`
            hidden_sizes: List of hidden layers sizes
            activation: Activation function
        """
        super().__init__(obs_space, action_space, head_factory, cnn_kind,
                         cnn_activation, linear_activation, dropout, **kwargs)
        obs_len = int(np.product(obs_space.shape))
        self.linear = DenseQRNN(self.linear[0].in_features, qrnn_hidden_size, qrnn_layers)
        self.head = head_factory(qrnn_hidden_size, self.pd)
        self.reset_weights()

    def forward(self, input, memory, done_flags):
        seq_len, batch_len = input.shape[:2]
        input = input.contiguous().view(seq_len * batch_len, *input.shape[2:])

        input = image_to_float(input)
        x = self._extract_features(input)

        x = x.view(seq_len, batch_len, -1)
        x, next_memory = self.linear(x, memory, done_flags)

        head = self.head(x)

        if self.do_log:
            self.logger.add_histogram('conv linear', x, self._step)

        return head, next_memory