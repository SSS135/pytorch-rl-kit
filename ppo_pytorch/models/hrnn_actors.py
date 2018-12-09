import gym
import gym.spaces
import numpy as np
import torch
import torch.nn as nn
from ..common.qrnn import DenseQRNN
from optfn.temporal_group_norm import TemporalGroupNorm
from torch.autograd import Variable

from .actors import ModularActor
from .cnn_actors import CNNActor
from .utils import image_to_float
from ..common.probability_distributions import make_pd, BernoulliPd, DiagGaussianPd, FixedStdGaussianPd, BetaPd
from .rnn_actors import RNNActor
from . import PolicyHead
from ..common.lstm import LSTM


class HRNNActor(ModularActor):
    def __init__(self, *args,
                 hidden_code_size=128, num_layers=3, h_action_size=16, rnn_kind='qrnn', **kwargs):
        super().__init__(*args, **kwargs)
        self.num_layers = num_layers
        self.hidden_code_size = hidden_code_size
        self.h_action_size = h_action_size

        def rnn_factory(n_in, n_out):
            if rnn_kind == 'qrnn':
                return DenseQRNN(n_in, n_out, num_layers, norm=self.norm)
            elif rnn_kind == 'lstm':
                return LSTM(n_in, n_out, self.num_layers)

        obs_len = int(np.product(self.observation_space.shape))

        self.h_pd = BetaPd(self.h_action_size, 1.01)
        self.gate_pd = BetaPd(1, 1.01)

        self.rnn_l1 = rnn_factory(obs_len, self.hidden_code_size)
        self.rnn_l2 = rnn_factory(obs_len, self.hidden_code_size)

        self.input_to_hidden_l2 = nn.Linear(self.hidden_code_size, self.hidden_code_size)
        self.input_to_hidden_l1 = nn.Linear(obs_len, self.hidden_code_size)
        self.input_to_state_l2 = nn.Linear(self.hidden_code_size, self.h_action_size)

        self.state_to_hidden_l2 = nn.Linear(self.h_action_size, self.hidden_code_size)
        self.state_to_hidden_l1 = nn.Linear(self.h_action_size, self.hidden_code_size)

        self.head_l2 = self._create_heads('head_l2', self.hidden_code_size, self.h_pd, self.head_factory)
        self.head_l1 = self._create_heads('head_l1', self.hidden_code_size, self.pd, self.head_factory)
        self.head_l2_gate = self._create_heads('head_l2_gate', self.hidden_code_size, self.gate_pd, self._gate_head_factory)

        self.reset_weights()

    def reset_weights(self):
        if not hasattr(self, 'head_l1'):
            return
        super().reset_weights()
        for head in [*self.head_l1.values(), *self.head_l2.values()]:
            head.reset_weights()

    @staticmethod
    def _gate_head_factory(hidden_size, pd):
        return dict(logits=PolicyHead(hidden_size, pd))

    def forward(self, input, memory, done_flags, action_l2=None, prev_action_l2=None, gate_l2=None):
        memory_l1, memory_l2 = memory.chunk(2, 0) if memory is not None else (None, None)

        hidden_l2, next_memory_l2 = self.rnn_l2(input, memory_l2, done_flags)
        state_l2 = self.input_to_state_l2(hidden_l2)
        hidden_l2 = self.input_to_hidden_l2(hidden_l2).sigmoid() * self.state_to_hidden_l2(state_l2)

        head_l2 = self._run_heads(hidden_l2, self.head_l2)
        if action_l2 is None:
            action_l2 = self.h_pd.sample(head_l2.logits)
            action_l2 /= action_l2.abs().max(-1, keepdim=True)[0]
            action_l2 = action_l2.detach()

        hidden_l1, next_memory_l1 = self.rnn_l1(input, memory_l1, done_flags)
        hidden_l1 = hidden_l1.sigmoid() * self.state_to_hidden_l1(action_l2)
        head_l1 = self._run_heads(hidden_l1, self.head_l1)

        next_memory = torch.cat([next_memory_l1, next_memory_l2], 0)

        return head_l1, head_l2, action_l2, state_l2, next_memory