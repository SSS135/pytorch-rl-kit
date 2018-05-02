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
from ..common.make_grid import make_grid
from ..common.probability_distributions import make_pd
from .actors import Actor, CNNActor
from optfn.qrnn import QRNN, DenseQRNN
from optfn.sigmoid_pow import sigmoid_pow
from optfn.rnn.dlstm import DLSTM
from pretrainedmodels import nasnetamobile
from optfn.shuffle_conv import ShuffleConv2d
from optfn.swish import Swish
from .heads import ActorCriticHead


class Sega_CNNSeqActor(CNNActor):
    def __init__(self, *args, seq_channels=512, **kwargs):
        super().__init__(*args, **kwargs)
        self.seq_channels = seq_channels
        del self.linear
        nf = 32
        in_c = self.observation_space.shape[0]
        self.convs = nn.ModuleList([
            self.make_layer(nn.Conv2d(in_c,   nf,     8, 4, 0, bias=self.norm is None)),
            self.make_layer(nn.Conv2d(nf,     nf * 2, 6, 3, 0, bias=self.norm is None)),
            self.make_layer(nn.Conv2d(nf * 2, nf * 4, 4, 2, 0, bias=self.norm is None)),
        ])
        layer_norm = self.norm is not None and 'layer' in self.norm
        self.seq_conv = nn.Sequential(
            nn.ReplicationPad1d((3, 0)),
            nn.Conv1d(1920, seq_channels, 4),
            *([nn.GroupNorm(1, seq_channels)] if layer_norm else []),
            nn.ReLU(),
            nn.ReplicationPad1d((3, 0)),
            nn.Conv1d(seq_channels, seq_channels, 4, bias=False),
            *([nn.GroupNorm(1, seq_channels)] if layer_norm else []),
            nn.ReLU(),
        )
        self.reset_weights()

    def forward(self, input, memory, done_flags):
        seq_len, batch_len = input.shape[:2]
        input = input.contiguous().view(seq_len * batch_len, *input.shape[2:])

        input = image_to_float(input)
        features = self._extract_features(input).view(seq_len, batch_len, -1)
        if memory is not None:
            features = torch.cat([memory, features], 0)
        # x = x.view(seq_len * batch_len, -1)
        # x = self.linear(x)
        # (B, C, S)
        x = features.permute(1, 2, 0).contiguous()
        x = self.seq_conv(x)
        # (S, B, C)
        x = x.permute(2, 0, 1).contiguous()

        if memory is not None:
            x = x[memory.shape[0]:]
        head = self.head(x)

        if self.do_log:
            self.logger.add_histogram('conv linear', x, self._step)

        return head, features[-1:]


class Sega_CNNHSeqActor(CNNActor):
    def __init__(self, *args, h_action_size=32, seq_channels=512, **kwargs):
        super().__init__(*args, **kwargs)
        self.seq_channels = seq_channels

        del self.linear

        self.h_action_space = gym.spaces.Box(-1, 1, h_action_size)
        self.h_observation_space = gym.spaces.Box(-1, 1, seq_channels)
        self.h_pd = make_pd(self.h_action_space)

        nf = 32
        in_c = self.observation_space.shape[0]
        self.convs = nn.ModuleList([
            self.make_layer(nn.Conv2d(in_c,   nf,     8, 4, 0, bias=self.norm is None)),
            self.make_layer(nn.Conv2d(nf,     nf * 2, 6, 3, 0, bias=self.norm is None)),
            self.make_layer(nn.Conv2d(nf * 2, nf * 4, 4, 2, 0, bias=self.norm is None)),
            self.make_layer(nn.Conv2d(nf * 4, nf * 8, 3, 1, 0, bias=self.norm is None)),
        ])
        self.l1_seq_conv = nn.Sequential(
            nn.ReplicationPad1d((3, 0)),
            nn.Conv1d(768, seq_channels, 4, bias=False),
            nn.GroupNorm(1, seq_channels),
            nn.ReLU(),
            nn.ReplicationPad1d((3, 0)),
            nn.Conv1d(seq_channels, seq_channels, 4, bias=False),
            nn.GroupNorm(1, seq_channels),
            nn.ReLU(),
        )
        self.l2_seq_conv = nn.Sequential(
            nn.ReplicationPad1d((3, 0)),
            nn.Conv1d(seq_channels, seq_channels, 4, bias=False),
            nn.GroupNorm(1, seq_channels),
            nn.ReLU(),
            nn.ReplicationPad1d((3, 0)),
            nn.Conv1d(seq_channels, seq_channels, 4, bias=False),
            nn.GroupNorm(1, seq_channels),
            nn.ReLU(),
        )
        self.l2_action_upsample = nn.Sequential(
            nn.Linear(h_action_size, seq_channels, bias=False),
            nn.GroupNorm(1, seq_channels),
            nn.ReLU(),
        )
        self.l1_merge = nn.Sequential(
            nn.Linear(seq_channels * 2, seq_channels, bias=False),
            nn.GroupNorm(1, seq_channels),
            nn.ReLU(),
        )

        self.head_l2 = ActorCriticHead(seq_channels, self.h_pd)
        self.reset_weights()

    def forward(self, input, memory, done_flags):
        seq_len, batch_len = input.shape[:2]
        input = input.contiguous().view(seq_len * batch_len, *input.shape[2:])

        input = image_to_float(input)
        features = self._extract_features(input).view(seq_len, batch_len, -1)
        if memory is not None:
            features = torch.cat([memory, features], 0)
        # (B, C, S)
        x = features.permute(1, 2, 0).contiguous()

        hidden_l1 = self.seq_conv(x)
        hidden_l2 = self.seq_conv(hidden_l1)
        # (S * B, C)
        hidden_l1 = hidden_l1.permute(2, 0, 1).contiguous().view(seq_len * batch_len, -1)
        hidden_l2 = hidden_l2.permute(2, 0, 1).contiguous().view(seq_len * batch_len, -1)
        head_l2 = self.head_l2(hidden_l2)
        action_l2 = self.h_pd.sample(head_l2.prob)
        action_l2_up = self.l2_action_upsample(action_l2)
        action_l2_dist = F.smooth_l1_loss(action_l2_up, hidden_l1)
        merged_l1 = self.l1_merge(torch.cat([hidden_l1, action_l2_up], 1))

        if memory is not None:
            x = x[memory.shape[0]:]
        head = self.head(x)

        if self.do_log:
            self.logger.add_histogram('conv linear', x, self._step)

        return head, features[-1:]