import copy
import pprint
from collections import namedtuple, OrderedDict, deque
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
from optfn.spectral_norm import spectral_norm
import random
from optfn.gadam import GAdam


class GanG(nn.Module):
    def __init__(self, state_size, noise_size, hidden_size=128):
        super().__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.noise_size = noise_size
        self.model = nn.Sequential(
            spectral_norm(nn.Linear(noise_size, hidden_size, bias=False)),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(True),
            spectral_norm(nn.Linear(hidden_size, hidden_size, bias=False)),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, state_size),
        )

    def forward(self, noise):
        return self.model(noise)


class GanD(nn.Module):
    def __init__(self, state_size, hidden_size=128):
        super().__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.model = nn.Sequential(
            spectral_norm(nn.Linear(state_size, hidden_size, bias=False)),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.1, True),
            spectral_norm(nn.Linear(hidden_size, hidden_size, bias=False)),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.1, True),
            spectral_norm(nn.Linear(hidden_size, 1)),
        )

    def forward(self, state):
        return self.model(state)


class PPO_GE(PPO):
    def __init__(self, *args,
                 density_buffer_size=16*1024,
                 state_buffer_size=16*1024,
                 gan_disc_optim_factory=partial(GAdam, lr=5e-4, betas=(0.0, 0.9), nesterov=0.75, amsgrad=True),
                 gan_gen_optim_factory=partial(GAdam, lr=1e-4, betas=(0.0, 0.9), nesterov=0.75, amsgrad=True),
                 gan_train_iters=8,
                 gan_batch_size=64,
                 noise_size=64,
                 min_surprise=0.8,
                 **kwargs):
        super().__init__(*args, **kwargs)
        assert state_buffer_size >= gan_train_iters * gan_batch_size
        self.density_buffer_size = density_buffer_size
        self.state_buffer_size = state_buffer_size
        self.gan_disc_optim_factory = gan_disc_optim_factory
        self.gan_gen_optim_factory = gan_gen_optim_factory
        self.gan_train_iters = gan_train_iters
        self.gan_batch_size = gan_batch_size
        self.noise_size = noise_size
        self.min_surprise = min_surprise

        self.gan_gen = GanG(self.model.hidden_code_size, noise_size)
        self.gan_disc = GanD(self.model.hidden_code_size)
        self.gan_gen_optim = gan_gen_optim_factory(self.gan_gen.parameters())
        self.gan_disc_optim = gan_disc_optim_factory(self.gan_disc.parameters())
        self.density_buffer = deque(maxlen=density_buffer_size)
        self.state_buffer = deque(maxlen=state_buffer_size)
        self.initial_gan_training_done = False

    def _ppo_update(self, data):
        self.gan_update(data)
        intrinsic_rewards = self.get_intrinsic_reward(data)
        adv = data.advantages
        # adv *= 0
        adv += intrinsic_rewards
        # data = data._replace(advantages=data.advantages + intrinsic_rewards)
        return super()._ppo_update(data)

    def get_intrinsic_reward(self, data):
        self.gan_gen = self.gan_gen.to(self.device_train).train()
        self.gan_disc = self.gan_disc.to(self.device_train).train()
        self.model = self.model.to(self.device_train).train()
        states = data.states.to(self.device_train)

        hidden_code = self.model(states).hidden_code

        if self._do_log:
            noise = torch.randn(states.shape[0], self.noise_size, device=self.device_train)
            self.logger.add_scalar('disc real', self.gan_disc(hidden_code).mean(), self.frame)
            self.logger.add_scalar('disc fake', self.gan_disc(self.gan_gen(noise)).mean(), self.frame)
            self.logger.add_scalar('disc rand', self.gan_disc(hidden_code.clone().normal_()).mean(), self.frame)

        disc_surprise = -self.gan_disc(hidden_code)
        disc_surprise = disc_surprise.squeeze().cpu().numpy()

        if len(self.density_buffer) == 0:
            self.density_buffer.extend(disc_surprise)
        density_idx = np.searchsorted(np.sort(self.density_buffer), disc_surprise)
        percentile_surprise = density_idx / len(self.density_buffer)
        rewards = (percentile_surprise - self.min_surprise).clip(min=0) / (1 - self.min_surprise)
        rewards = rewards ** 2
        rewards = torch.tensor(rewards, dtype=torch.float)

        returns = calc_returns(rewards.view(-1, self.num_actors),
                               torch.zeros(self.horizon + 1, self.num_actors),
                               data.dones.view(-1, self.num_actors), 0.9)
        returns = returns.view(-1)
        returns = (returns - returns.mean()) / (returns.std() + 1e-3)

        self.density_buffer.extend(disc_surprise)
        return returns

    def gan_update(self, data):
        # move model to cuda or cpu
        self.gan_gen = self.gan_gen.to(self.device_train).train()
        self.gan_disc = self.gan_disc.to(self.device_train).train()
        self.model = self.model.to(self.device_train).train()

        self.state_buffer.extend(data.states)
        if len(self.state_buffer) != self.state_buffer_size:
            return
        if not self.initial_gan_training_done:
            for i in range(50):
                self.gan_train()
            self.initial_gan_training_done = True
            for i in range(50):
                self.gan_train()
        else:
            self.gan_train()

    def gan_train(self):
        states = random.sample(self.state_buffer, self.gan_batch_size * self.gan_train_iters)
        states = torch.stack(states, 0)
        data = (states.pin_memory() if self.device_train.type == 'cuda' else states,)

        for batch_num in range(self.gan_train_iters):
            st, = [x[batch_num * self.gan_batch_size: (batch_num + 1) * self.gan_batch_size]
                      .to(self.device_train) for x in data]
            # if ppo_iter == self.ppo_iters - 1 and loader_iter == 0:
            #     self.model.set_log(self.logger, self._do_log, self.step)

            hidden_code = self.model(st).hidden_code

            # disc real
            with torch.enable_grad():
                disc_real = self.gan_disc(hidden_code)
                real_loss = -disc_real.clamp(max=1).mean()
            real_loss.backward()
            self.gan_disc_optim.step()
            self.gan_gen_optim.zero_grad()
            self.gan_disc_optim.zero_grad()

            # disc fake
            noise = torch.randn(self.gan_batch_size, self.noise_size, device=self.device_train)
            with torch.enable_grad():
                gen_states = self.gan_gen(noise)
                disc_fake = self.gan_disc(gen_states.detach())
                fake_loss = disc_fake.clamp(min=-1).mean()
            fake_loss.backward()
            self.gan_disc_optim.step()
            self.gan_gen_optim.zero_grad()
            self.gan_disc_optim.zero_grad()

            if not self.initial_gan_training_done:
                continue

            # gen
            with torch.enable_grad():
                disc_fake = self.gan_disc(gen_states)
                gen_loss = -disc_fake.mean()
            gen_loss.backward()
            self.gan_gen_optim.step()
            self.gan_gen_optim.zero_grad()
            self.gan_disc_optim.zero_grad()

        # print(real_loss.mean().item(), fake_loss.mean().item(), gen_loss.mean().item())

            # self.model.set_log(self.logger, False, self.step)
