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
import random


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
        self.states = None
        self.actions = None
        self.rewards = None
        self.dones = None
        self.index = 0
        self.full_loop = False

    def push(self, states, actions, rewards, dones):
        states = np.asarray(states)
        actions = np.asarray(actions)
        rewards = np.asarray(rewards)
        dones = np.asarray(dones)

        assert states.ndim >= 3
        assert actions.ndim >= 3
        assert rewards.shape == dones.shape and rewards.ndim == 2
        assert rewards.shape == states.shape[:2] and rewards.shape == actions.shape[:2]

        if self.states is None:
            actors = states.shape[1]
            self.states = np.zeros((self.capacity, actors, *states.shape[2:]), dtype=states.dtype)
            self.actions = np.zeros((self.capacity, actors, *actions.shape[2:]), dtype=actions.dtype)
            self.rewards = np.zeros((self.capacity, actors), dtype=np.float32)
            self.dones = np.zeros((self.capacity, actors), dtype=np.bool)

        if self.index + states.shape[0] <= self.capacity:
            self._push_unchecked(states, actions, rewards, dones)
        else:
            n = self.capacity - self.index - states.shape[0]
            self._push_unchecked(states[:n], actions[:n], rewards[:n], dones[:n])
            self.index = 0
            self._push_unchecked(states[n:], actions[n:], rewards[n:], dones[n:])
            self.full_loop = True

    def _push_unchecked(self, states, actions, rewards, dones):
        a = self.index
        b = self.index + states.shape[0]
        self.states[a: b] = states
        self.actions[a: b] = actions
        self.rewards[a: b] = rewards
        self.dones[a: b] = dones
        self.index += states.shape[0]

    def sample(self, rollouts, horizon):
        states = np.zeros((horizon, rollouts, *self.states.shape[2:]), dtype=self.states.dtype)
        actions = np.zeros((horizon, rollouts, *self.actions.shape[2:]), dtype=self.actions.dtype)
        rewards = np.zeros((horizon, rollouts), dtype=self.rewards.dtype)
        dones = np.zeros((horizon, rollouts), dtype=self.dones.dtype)

        for ri in range(rollouts):
            rand_r = np.random.randint(self.states.shape[1])
            rand_h = np.random.randint((self.capacity if self.full_loop else self.index) - horizon)
            src_slice = (slice(rand_h, rand_h + horizon), rand_r)
            dst_slice = (slice(None, None), ri)
            states[dst_slice] = self.states[src_slice]
            actions[dst_slice] = self.actions[src_slice]
            rewards[dst_slice] = self.rewards[src_slice]
            dones[dst_slice] = self.dones[src_slice]

        return states, actions, rewards, dones

    def __len__(self):
        return min(self.index, self.capacity)


class MPPO(PPO):
    def __init__(self, *args,
                 replay_buffer_size=100_000,
                 world_optim_factory=partial(optim.Adam, lr=5e-4),
                 world_train_rollouts=16,
                 world_train_horizon=16,
                 world_batch_size=64,
                 **kwargs):
        super().__init__(*args, **kwargs)
        assert world_batch_size % world_train_horizon == 0 and \
               (world_train_rollouts * world_train_horizon) % world_batch_size == 0
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

        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.float)
        rewards = torch.tensor(rewards, dtype=torch.float)
        dones = torch.tensor(dones, dtype=torch.float)

