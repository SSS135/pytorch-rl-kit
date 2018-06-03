import copy
import pprint
from collections import namedtuple, OrderedDict
from functools import partial

import gym.spaces
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from ..common import DecayLR, ValueDecay
from ..common.gae import calc_advantages, calc_returns
from ..common.multi_dataset import MultiDataset
from ..common.probability_distributions import DiagGaussianPd
from ..common.rl_base import RLBase
from ..models import QRNNActorCritic
from ..models.heads import HeadOutput
from .ppo import PPO, TrainingData
from collections import namedtuple
import torch.nn as nn


class EnvGenerator(nn.Module):
    def __init__(self, state_size, action_pd, hidden_size=128):
        super().__init__()
        self.state_size = state_size
        self.action_pd = action_pd
        self.hidden_size = hidden_size
        self.action_embedding = nn.Linear(action_pd.input_vector_len, hidden_size)
        self.state_embedding = nn.Linear(state_size, hidden_size)
        self.model = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, state_size * 3 + 2),
        )

    def forward(self, cur_states, actions):
        action_inputs = self.action_pd.to_inputs(actions)
        action_emb = self.action_embedding(action_inputs)
        state_emb = self.state_embedding(cur_states)
        out = self.model(action_emb + state_emb)
        next_states, forget_gate, input_gate = out[..., :-2].chunk(3, dim=-1)
        forget_gate, input_gate = forget_gate.sigmoid(), input_gate.sigmoid()
        next_states = forget_gate * cur_states + input_gate * next_states
        rewards, dones = out[..., -2], out[..., -1]
        return next_states, rewards, dones


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = None
        self.index = 0

    def push(self, states, actions, rewards, dones):
        raise NotImplementedError

    def sample(self, rollouts, horizon):
        raise NotImplementedError

    def __len__(self):
        return min(self.index, self.capacity)


class MPPO(PPO):
    def __init__(self, *args,
                 replay_buffer_size=100_000,
                 world_optim_factory=partial(optim.Adam, lr=5e-4),
                 world_train_rollouts=16,
                 world_train_horizon=16,
                 world_train_iters=16,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.replay_buffer_size = replay_buffer_size
        self.world_optim_factory = world_optim_factory
        self.world_train_rollouts = world_train_rollouts
        self.world_train_horizon = world_train_horizon
        self.world_train_iters = world_train_iters

        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.world_model = EnvGenerator(self.model.hidden_code_size, self.model.pd)
        self.world_optim = world_optim_factory(self.world_model.parameters())
        self.prev_actions = None

    def _train(self):
        self._update_replay_buffer()
        self._train_world()
        return super()._train()

    def _update_replay_buffer(self):
        # H x B x *
        self.replay_buffer.push(self.sample.states[:-1], self.sample.actions[:-1],
                                self.sample.rewards, self.sample.dones)

    def _train_world(self):
        # H x B x *
        states, actions, rewards, dones = self.replay_buffer.sample(
            self.world_train_rollouts * self.world_train_iters, self.world_train_horizon)

