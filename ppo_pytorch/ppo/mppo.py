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
    def __init__(self, state_size, action_pd, hidden_size=256, action_embedding_size=64):
        super().__init__()
        self.state_size = state_size
        self.action_pd = action_pd
        self.hidden_size = hidden_size
        self.action_embedding_size = action_embedding_size
        self.action_embedding = nn.Sequential(
            nn.Linear(action_pd.input_vector_len, action_embedding_size),
            nn.ReLU(True),
        )
        self.model = nn.Sequential(
            nn.Linear(state_size + action_embedding_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, state_size * 3 + 2),
        )

    def forward(self, cur_states, actions):
        action_inputs = self.action_pd.to_inputs(actions)
        action_emb = self.action_embedding(action_inputs)
        model_input = torch.cat([cur_states, action_emb], -1)
        out = self.model(model_input)
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
        pass

    def sample(self, rollout_count, rollout_steps):
        pass

    def __len__(self):
        return min(self.index, self.capacity)


class MPPO(PPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _step(self, prev_states, rewards, dones, cur_states) -> np.ndarray:
        return super()._step(prev_states, rewards, dones, cur_states)

    def _train(self):
        return super()._train()

