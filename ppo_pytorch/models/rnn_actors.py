import gym
import gym.spaces
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .actors import Actor
from ..common.qrnn import DenseQRNN
from ..common.lstm import LSTM


class RNNActor(Actor):
    def __init__(self, observation_space: gym.Space, action_space: gym.Space, *args,
                 hidden_code_size=128, num_layers=3, rnn_kind='qrnn', **kwargs):
        """
        Args:
            observation_space: Env's observation space
            action_space: Env's action space
            head_factory: Function which accept (hidden vector size, `ProbabilityDistribution`) and return `HeadBase`
            hidden_code_size: Hidden layer width
            activation: Activation function
        """
        super().__init__(observation_space, action_space, *args, **kwargs)
        self.num_layers = num_layers
        self.hidden_code_size = hidden_code_size
        self.rnn_kind = rnn_kind
        obs_len = int(np.product(observation_space.shape))
        if rnn_kind == 'qrnn':
            self.rnn = DenseQRNN(obs_len, hidden_code_size, num_layers, norm=self.norm)
        elif rnn_kind == 'lstm':
            self.rnn = LSTM(obs_len, self.hidden_code_size, self.num_layers)
        self._init_heads(self.hidden_code_size)
        self.reset_weights()

    def forward(self, input, memory, done_flags):
        x, next_memory = self.rnn(input, memory, done_flags)
        head = self._run_heads(x)
        head.hidden_code = x
        return head, next_memory