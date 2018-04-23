import math
from functools import partial
from typing import Optional, List, Callable

import gym
import gym.spaces
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
from optfn.sigmoid_pow import sigmoid_pow
from optfn.rnn.dlstm import DLSTM
from pretrainedmodels import nasnetamobile
from optfn.shuffle_conv import ShuffleConv2d
from optfn.swish import Swish
from .heads import ActorCriticHead


class QRNNActor(Actor):
    def __init__(self, observation_space: gym.Space, action_space: gym.Space, head_factory: Callable,
                 hidden_size=128, num_layers=3, **kwargs):
        """
        Args:
            observation_space: Env's observation space
            action_space: Env's action space
            head_factory: Function which accept (hidden vector size, `ProbabilityDistribution`) and return `HeadBase`
            hidden_sizes: List of hidden layers sizes
            activation: Activation function
        """
        super().__init__(observation_space, action_space, **kwargs)
        obs_len = int(np.product(observation_space.shape))
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
    def __init__(self, *args, qrnn_hidden_size=512, qrnn_layers=3, **kwargs):
        """
        Args:
            observation_space: Env's observation space
            action_space: Env's action space
            head_factory: Function which accept (hidden vector size, `ProbabilityDistribution`) and return `HeadBase`
            hidden_sizes: List of hidden layers sizes
            activation: Activation function
        """
        super().__init__(*args, **kwargs)
        self.qrnn_hidden_size = qrnn_hidden_size
        self.qrnn_layers = qrnn_layers
        del self.linear
        assert self.cnn_kind == 'large' # custom (2,066,432 parameters)
        nf = 32
        self.convs = nn.ModuleList([
            self.make_layer(nn.Conv2d(self.observation_space.shape[0], nf, 4, 2, 0, bias=False)),
            nn.MaxPool2d(3, 2),
            self.make_layer(nn.Conv2d(nf, nf * 2, 4, 2, 0, bias=False)),
            self.make_layer(nn.Conv2d(nf * 2, nf * 4, 4, 2, 1, bias=False)),
            self.make_layer(nn.Conv2d(nf * 4, nf * 8, 4, 2, 1, bias=False)),
        ])
        # self.linear = self.make_layer(nn.Linear(1024, 512))
        self.qrnn = DenseQRNN(512, qrnn_hidden_size, qrnn_layers)
        self.head = self.head_factory(qrnn_hidden_size, self.pd)
        self.reset_weights()

    def forward(self, input, memory, done_flags):
        seq_len, batch_len = input.shape[:2]
        input = input.contiguous().view(seq_len * batch_len, *input.shape[2:])

        input = image_to_float(input)
        x = self._extract_features(input)
        # x = x.view(seq_len * batch_len, -1)
        # x = self.linear(x)
        x = x.view(seq_len, batch_len, -1)
        x, next_memory = self.qrnn(x, memory, done_flags)

        head = self.head(x)

        if self.do_log:
            self.logger.add_histogram('conv linear', x, self._step)

        return head, next_memory


class Sega_CNN_QRNNActor(CNN_QRNNActor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nf = 32
        in_c = self.observation_space.shape[0]
        self.convs = nn.ModuleList([
            self.make_layer(nn.Conv2d(in_c,   nf    , 4, 2, 0), allow_norm=False),
            self.make_layer(nn.Conv2d(nf    , nf * 2, 4, 2, 0, bias=self.norm is None)),
            self.make_layer(nn.Conv2d(nf * 2, nf * 4, 4, 2, 0, bias=self.norm is None)),
            self.make_layer(nn.Conv2d(nf * 4, nf * 4, 4, 2, 0, bias=self.norm is None)),
            self.make_layer(nn.Conv2d(nf * 4, nf * 4, 3, 1, 0, bias=self.norm is None)),
        ])
        # self.linear = self.make_layer(nn.Linear(1536, 512, bias='layer' not in self.norm))
        self.qrnn = DLSTM(2304, self.qrnn_hidden_size, self.qrnn_layers)
        self.reset_weights()


class Sega_CNN_HQRNNActor(Sega_CNN_QRNNActor):
    def __init__(self, h_action_size=64, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.h_action_size = h_action_size

        del self.qrnn

        self.h_action_space = gym.spaces.Box(-1, 1, h_action_size)
        self.h_observation_space = gym.spaces.Box(-1, 1, self.qrnn_hidden_size)
        self.h_pd = make_pd(self.h_action_space)

        self.l1_qrnn = DenseQRNN(768, self.qrnn_hidden_size, self.qrnn_layers)
        self.l2_qrnn = DenseQRNN(self.qrnn_hidden_size, self.qrnn_hidden_size, self.qrnn_layers)
        self.l2_action_upsample = nn.Sequential(
            nn.Linear(h_action_size, self.qrnn_hidden_size, bias=False),
            LayerNorm1d(self.qrnn_hidden_size),
            nn.ReLU(),
        )
        self.l1_merge = nn.Sequential(
            nn.Linear(self.qrnn_hidden_size * 2, self.qrnn_hidden_size, bias=False),
            LayerNorm1d(self.qrnn_hidden_size),
            nn.ReLU(),
        )
        self.l2_head = ActorCriticHead(self.qrnn_hidden_size, self.h_pd)
        self.reset_weights()

    def forward(self, input, memory, done_flags):
        seq_len, batch_len = input.shape[:2]
        input = input.contiguous().view(seq_len * batch_len, *input.shape[2:])

        input = image_to_float(input)
        x = self._extract_features(input)
        # x = x.view(seq_len * batch_len, -1)
        # x = self.linear(x)
        x = x.view(seq_len, batch_len, -1)
        memory = memory.chunk(2, 0)
        hidden_1, next_memory_1 = self.l1_qrnn(x, memory[0], done_flags)
        hidden_2, next_memory_2 = self.l2_qrnn(hidden_1, memory[1], done_flags)

        head = self.head(x)

        return head, next_memory


class HLevel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.qrnn = DenseQRNN(input_size, hidden_size, num_layers)