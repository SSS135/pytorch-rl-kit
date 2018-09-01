import gym
import gym.spaces
import numpy as np
import torch
import torch.nn as nn
from ..common.qrnn import DenseQRNN

from .actors import Actor
from .cnn_actors import CNNActor
from .utils import image_to_float


class QRNNActor(Actor):
    def __init__(self, observation_space: gym.Space, action_space: gym.Space, *args,
                 qrnn_hidden_size=128, qrnn_layers=3, **kwargs):
        """
        Args:
            observation_space: Env's observation space
            action_space: Env's action space
            head_factory: Function which accept (hidden vector size, `ProbabilityDistribution`) and return `HeadBase`
            hidden_sizes: List of hidden layers sizes
            activation: Activation function
        """
        super().__init__(observation_space, action_space, *args, **kwargs)
        self.qrnn_hidden_size = self.hidden_code_size = qrnn_hidden_size
        self.qrnn_layers = qrnn_layers
        obs_len = int(np.product(observation_space.shape))
        self.qrnn = DenseQRNN(obs_len, qrnn_hidden_size, qrnn_layers, norm=self.norm)
        self._init_heads(self.hidden_code_size)
        self.reset_weights()

    def forward(self, input, memory, done_flags):
        x, next_memory = self.qrnn(input, memory, done_flags)
        head = self._run_heads(x)
        head.hidden_code = x
        return head, next_memory


class CNN_QRNNActor(CNNActor):
    def __init__(self, *args, qrnn_hidden_size=512, qrnn_layers=2, qrnn_norm=None, **kwargs):
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
        # self.linear = self.make_layer(nn.Linear(1024, 512))
        self.qrnn = DenseQRNN(self.linear[0].in_features, qrnn_hidden_size, qrnn_layers, norm=qrnn_norm)
        del self.linear
        self.hidden_code_size = qrnn_hidden_size
        self._init_heads(self.hidden_code_size)
        self.reset_weights()

    def forward(self, input, memory, done_flags):
        seq_len, batch_len = input.shape[:2]
        input = input.contiguous().view(seq_len * batch_len, *input.shape[2:])

        input = image_to_float(input)
        x = self._extract_features(input)
        x = x.view(seq_len, batch_len, -1)
        x, next_memory = self.qrnn(x, memory, done_flags)

        head = self._run_heads(x)
        head.hidden_code = x

        if self.do_log:
            self.logger.add_histogram('conv linear', x, self._step)

        return head, next_memory
