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
from optfn.spectral_norm import spectral_norm
from optfn.gadam import GAdam
from collections import deque
from ..models.utils import weights_init
from optfn.grad_running_norm import GradRunningNorm
from optfn.drrelu import DRReLU


def spectral_init(module, gain=1):
    nn.init.kaiming_uniform_(module.weight, gain)
    if module.bias is not None:
        module.bias.data.zero_()
    return spectral_norm(module)


class GanG(nn.Module):
    def __init__(self, state_size, action_pd, hidden_size=256):
        super().__init__()
        self.state_size = state_size
        self.action_pd = action_pd
        self.hidden_size = hidden_size
        self.action_embedding = spectral_init(nn.Linear(action_pd.input_vector_len, hidden_size, bias=False))
        self.state_embedding = spectral_init(nn.Linear(state_size, hidden_size, bias=False))
        self.model = nn.Sequential(
            nn.GroupNorm(4, hidden_size),
            DRReLU(-0.3, 0.3, 0.95, 1.05, True),
            spectral_init(nn.Linear(hidden_size, hidden_size, bias=False)),
            nn.GroupNorm(4, hidden_size),
            DRReLU(-0.3, 0.3, 0.95, 1.05, True),
            # spectral_norm(nn.Linear(hidden_size, hidden_size)),
            # nn.GroupNorm(4, hidden_size),
            # nn.ReLU(inplace=True),
            nn.Linear(hidden_size, state_size * 3 + 2),
        )

    def forward(self, cur_states, actions):
        action_inputs = self.action_pd.to_inputs(actions)
        action_emb = self.action_embedding(action_inputs)
        state_emb = self.state_embedding(cur_states)
        out = self.model(action_emb + state_emb)
        next_states, forget_gate, input_gate, rewards, dones = \
            out.split([self.state_size, self.state_size, self.state_size, 1, 1], dim=-1)
        # next_states = (next_states - next_states.mean(-1, keepdim=True)) / next_states.std(-1, keepdim=True)
        # next_states = next_states * cur_states.std(-1, keepdim=True) + cur_states.mean(-1, keepdim=True)
        next_states = (forget_gate * 0.5 + 0.5) * cur_states + (input_gate * 0.5 + 0.5) * next_states
        # next_states, rewards, dones = out.split([self.state_size, 1, 1], dim=-1)
        # next_states = cur_states + next_states
        dones, rewards = dones.squeeze(-1), rewards.squeeze(-1)
        dones_mask = (dones > 0).detach()
        next_states[dones_mask] = cur_states[dones_mask]
        return next_states, rewards, dones


class GanD(nn.Module):
    def __init__(self, state_size, action_pd, hidden_size=256):
        super().__init__()
        self.state_size = state_size
        self.action_pd = action_pd
        self.hidden_size = hidden_size
        self.action_embedding = spectral_init(nn.Linear(action_pd.input_vector_len, hidden_size, bias=False))
        self.cur_state_embedding = spectral_init(nn.Linear(state_size, hidden_size, bias=False))
        self.next_state_embedding = spectral_init(nn.Linear(state_size, hidden_size, bias=False))
        self.reward_done_embedding = spectral_init(nn.Linear(2, hidden_size, bias=False))
        self.model_start = nn.Sequential(
            nn.GroupNorm(4, hidden_size),
            DRReLU(-0.3, 0.3, 0.95, 1.05, True),
            spectral_init(nn.Linear(hidden_size, hidden_size, bias=False)),
            # spectral_norm(nn.Linear(hidden_size * 2, hidden_size)),
            # nn.GroupNorm(4, hidden_size),
            # nn.ReLU(inplace=True),
        )
        self.model_end = nn.Sequential(
            nn.GroupNorm(4, hidden_size),
            DRReLU(-0.3, 0.3, 0.95, 1.05, True),
            spectral_init(nn.Linear(hidden_size, 1)),
        )

    def forward(self, cur_states, next_states, actions, rewards, dones):
        action_inputs = self.action_pd.to_inputs(actions)
        action_emb = self.action_embedding(action_inputs)
        cur_state_emb = self.cur_state_embedding(cur_states)
        next_state_emb = self.next_state_embedding(next_states)
        # print(next_state_emb[dones > 0].shape)
        next_state_emb[(dones > 0).detach()] = 0 # *= 1 - (dones.unsqueeze(-1) > 0.5).float().detach()
        reward_done_emb = self.reward_done_embedding(torch.stack([rewards, dones], -1))
        features = self.model_start(action_emb + cur_state_emb + next_state_emb + reward_done_emb)
        out = self.model_end(features).squeeze(-1)
        return out, features


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
            self.dones = np.zeros((self.capacity, actors), dtype=np.uint8)

        if self.index + states.shape[0] <= self.capacity:
            self._push_unchecked(states, actions, rewards, dones)
        else:
            n = self.capacity - self.index
            self._push_unchecked(states[:n], actions[:n], rewards[:n], dones[:n])
            self.index = 0
            self.full_loop = True
            self._push_unchecked(states[n:], actions[n:], rewards[n:], dones[n:])

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
            rand_h = np.random.randint(len(self) - horizon)
            src_slice = (slice(rand_h, rand_h + horizon), rand_r)
            dst_slice = (slice(None), ri)
            states[dst_slice] = self.states[src_slice]
            actions[dst_slice] = self.actions[src_slice]
            rewards[dst_slice] = self.rewards[src_slice]
            dones[dst_slice] = self.dones[src_slice]

        return states, actions, rewards, dones

    def __len__(self):
        return self.capacity if self.full_loop else self.index


class MPPO(PPO):
    def __init__(self, *args,
                 density_buffer_size=16 * 1024,
                 replay_buffer_size=64 * 1024,
                 world_disc_optim_factory=partial(GAdam, lr=4e-4, betas=(0.5, 0.99), weight_decay=1e-4),
                 world_gen_optim_factory=partial(GAdam, lr=1e-4, betas=(0.5, 0.99), weight_decay=1e-4),
                 world_train_iters=8,
                 world_train_rollouts=128,
                 world_train_horizon=4,
                 **kwargs):
        super().__init__(*args, **kwargs)
        # assert world_batch_size % world_train_horizon == 0 and \
        #        (world_train_rollouts * world_train_horizon) % world_batch_size == 0
        assert replay_buffer_size >= world_train_iters * world_train_rollouts * (world_train_horizon + 1)

        self.density_buffer_size = density_buffer_size
        self.replay_buffer_size = replay_buffer_size
        self.world_disc_optim_factory = world_disc_optim_factory
        self.world_gen_optim_factory = world_gen_optim_factory
        self.world_train_iters = world_train_iters
        # self.world_batch_size = world_batch_size
        self.world_train_rollouts = world_train_rollouts
        self.world_train_horizon = world_train_horizon

        self.world_gen = GanG(self.model.hidden_code_size, self.model.pd)
        self.world_disc = GanD(self.model.hidden_code_size, self.model.pd)
        self.world_gen_optim = world_gen_optim_factory(self.world_gen.parameters())
        self.world_disc_optim = world_disc_optim_factory(self.world_disc.parameters())
        self.density_buffer = deque(maxlen=density_buffer_size)
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.initial_world_training_done = False
        # self.cur_state_gn = GradRunningNorm(momentum=0.5)
        # self.next_state_gn = GradRunningNorm(momentum=0.5)
        # self.reward_gn = GradRunningNorm(momentum=0.5)
        # self.done_gn = GradRunningNorm(momentum=0.5)

    def _ppo_update(self, data):
        self._update_replay_buffer(data)
        min_buf_size = self.world_train_iters * self.world_train_rollouts * (self.world_train_horizon + 1)
        if len(self.replay_buffer) >= min_buf_size or len(self.replay_buffer) == self.replay_buffer_size:
            if self.initial_world_training_done:
                self._train_world()
            else:
                # for _ in range(50):
                #     self._train_world()
                self.initial_world_training_done = True
                # for _ in range(10):
                #     self._train_world()
        return super()._ppo_update(data)

    def _update_replay_buffer(self, data):
        # H x B x *
        self.replay_buffer.push(
            data.states.view(-1, self.num_actors, *data.states.shape[1:]),
            data.actions.view(-1, self.num_actors, *data.actions.shape[1:]),
            data.rewards.view(-1, self.num_actors, *data.rewards.shape[1:]),
            data.dones.view(-1, self.num_actors, *data.dones.shape[1:])
        )

    def _train_world(self):
        # move model to cuda or cpu
        self.world_gen = self.world_gen.to(self.device_train).train()
        self.world_disc = self.world_disc.to(self.device_train).train()
        self.model = self.model.to(self.device_train).train()

        # (H, B, ...)
        all_states, all_actions, all_rewards, all_dones = self.replay_buffer.sample(
            self.world_train_rollouts * self.world_train_iters, self.world_train_horizon + 1)

        all_dones = all_dones.astype(np.float32) * 2 - 1
        # all_dones += 0.02 * np.random.randn(*all_dones.shape)
        # all_rewards += 0.02 * np.random.randn(*all_rewards.shape)

        data = [torch.from_numpy(x) for x in (all_states, all_actions, all_rewards, all_dones)]
        if self.device_train.type == 'cuda':
            data = [x.pin_memory() for x in data]

        for train_iter in range(self.world_train_iters):
            slc = (slice(None), slice(train_iter * self.world_train_rollouts, (train_iter + 1) * self.world_train_rollouts))
            # (H, B, ...)
            states, actions, rewards, dones = [x[slc].to(self.device_train) for x in data]

            # disc real
            hidden_codes = self.model(states.view(-1, *states.shape[2:]), only_hidden_code_output=True).hidden_code
            hidden_codes = hidden_codes.view(*states.shape[:2], *hidden_codes.shape[1:])
            # hidden_codes += 0.05 * torch.randn_like(hidden_codes)

            # with torch.enable_grad():
            #     gen_next_code, gen_rewards, gen_dones = self.world_gen(hidden_codes[0], actions[0])
            #     loss = F.mse_loss(gen_next_code, hidden_codes[1]) + \
            #            F.mse_loss(gen_rewards, rewards[0]) + \
            #            F.mse_loss(gen_dones, dones[0])
            # loss.backward()
            # self.world_gen_optim.step()
            # self.world_disc_optim.zero_grad()
            # self.world_gen_optim.zero_grad()

            with torch.enable_grad():
                # disc_real = self.world_disc(
                #     hidden_codes[0],
                #     hidden_codes[1],
                #     actions[0],
                #     rewards[0],
                #     dones[0],
                # )
                disc_real, features_real = self.world_disc(
                    hidden_codes[:-1].view(-1, *hidden_codes.shape[2:]),
                    hidden_codes[1:].view(-1, *hidden_codes.shape[2:]),
                    actions[:-1].view(-1, *actions.shape[2:]),
                    rewards[:-1].view(-1, *rewards.shape[2:]),
                    dones[:-1].view(-1, *dones.shape[2:])
                )
                # disc_real = [
                #     self.world_disc(hidden_codes[i], hidden_codes[i + 1],
                #                     actions[i], rewards[i], dones[i].clamp(0.1, 0.9))
                #     for i in range(self.world_train_horizon - 1)
                # ]
                # # (H * B)
                # disc_real = torch.cat(disc_real, dim=0)
                # mask = self.get_done_mask(dones[:-1]).view(-1)
                real_loss = -disc_real.clamp(max=1).mean()
                # disc_real = (disc_real - disc_real.mean()) / disc_real.var().add(1e-8).sqrt()
                # real_loss = (1 - disc_real.clamp(max=1)).pow(2).mean()
            real_loss.backward()
            self.world_disc_optim.step()
            self.world_disc_optim.zero_grad()
            self.world_gen_optim.zero_grad()

            # disc fake
            disc_fake = []
            all_gen_hidden_codes = [hidden_codes[0]]
            all_gen_actions = []
            all_gen_rewards = []
            all_gen_dones = []
            for i in range(self.world_train_horizon):
                ac_out = self.model(all_gen_hidden_codes[-1], hidden_code_input=True)
                cur_code = all_gen_hidden_codes[-1]
                cur_actions = self.model.pd.sample(ac_out.probs)
                with torch.enable_grad():
                    gen_next_code, gen_rewards, gen_dones = self.world_gen(cur_code, cur_actions)
                    d, _ = self.world_disc(cur_code.detach(), gen_next_code.detach(),
                                           cur_actions, gen_rewards.detach(), gen_dones.detach())
                all_gen_hidden_codes.append(gen_next_code)
                all_gen_actions.append(cur_actions)
                all_gen_rewards.append(gen_rewards)
                all_gen_dones.append(gen_dones)
                disc_fake.append(d)
            with torch.enable_grad():
                # gen_next_code, gen_rewards, gen_dones = self.world_gen(hidden_codes[0], actions[0])
                # disc_fake = self.world_disc(hidden_codes[0].detach(), gen_next_code.detach(),
                #                             actions[0], gen_rewards.detach(), gen_dones.detach())

                all_gen_hidden_codes = torch.stack(all_gen_hidden_codes, 0)
                all_gen_actions = torch.stack(all_gen_actions, 0)
                all_gen_rewards = torch.stack(all_gen_rewards, 0)
                all_gen_dones = torch.stack(all_gen_dones, 0)
                disc_fake = torch.stack(disc_fake, 0)

                # mask = self.get_done_mask(all_gen_dones).view(-1)
                fake_loss = disc_fake.clamp(min=-1).mean()
                # disc_fake = (disc_fake - disc_fake.mean()) / disc_fake.var().add(1e-8).sqrt()
                # fake_loss = (-1 - disc_fake.clamp(min=-1)).pow(2).mean()
            fake_loss.backward()
            self.world_disc_optim.step()
            self.world_disc_optim.zero_grad()
            self.world_gen_optim.zero_grad()

            if not self.initial_world_training_done:
                continue

            # gen
            with torch.enable_grad():
                # disc_gen = self.world_disc(hidden_codes[0], gen_next_code, actions[0], gen_rewards, gen_dones)

                disc_gen, features_gen = self.world_disc(
                    all_gen_hidden_codes[:-1].view(-1, *hidden_codes.shape[2:]),
                    all_gen_hidden_codes[1:].view(-1, *hidden_codes.shape[2:]),
                    all_gen_actions.view(-1, *actions.shape[2:]),
                    all_gen_rewards.view(-1, *rewards.shape[2:]),
                    all_gen_dones.view(-1, *dones.shape[2:])
                )

                # disc_gen = [
                #     self.world_disc(all_gen_hidden_codes[i], all_gen_hidden_codes[i + 1],
                #                     all_gen_actions[i], all_gen_rewards[i], all_gen_dones[i])
                #     for i in range(self.world_train_horizon)
                # ]
                # # (H * B)
                # disc_gen = torch.cat(disc_gen, dim=0)
                gen_loss = -disc_gen.mean() #+ \
                           #(features_gen.mean(0) - features_real.mean(0).detach()).pow(2).mean() #+ \
                           #(features_gen.std(0) - features_real.std(0).detach()).pow(2).mean()
                # gen_loss = (1 - disc_gen.clamp(max=1)).pow(2).mean()
            gen_loss.backward()
            self.world_gen_optim.step()
            self.world_disc_optim.zero_grad()
            self.world_gen_optim.zero_grad()

        if self._do_log:
            gen_next_code, gen_rewards, gen_dones = self.world_gen(hidden_codes[0], actions[0])
            rmse = lambda a, b: (a - b).abs().mean().item()
            self.logger.add_scalar('gen 1-step state err', rmse(gen_next_code, hidden_codes[1]), self.frame)
            state_norm_rmse = rmse(gen_next_code, hidden_codes[1]) / rmse(hidden_codes[0], hidden_codes[1])
            self.logger.add_scalar('gen 1-step state norm err', state_norm_rmse, self.frame)
            self.logger.add_scalar('gen 1-step reward err', rmse(gen_rewards, rewards[0]), self.frame)
            self.logger.add_scalar('gen 1-step done err', rmse(gen_dones, dones[0]), self.frame)

    def get_done_mask(self, dones):
        done_mask = (dones > 0).cpu().numpy().copy()
        done_max = np.zeros(done_mask.shape[1], dtype=np.uint8)
        for i in range(done_mask.shape[0]):
            new_done_max = np.maximum(done_max, done_mask[i])
            done_mask[i] = new_done_max & (~done_mask[i] | done_max)
            done_max = new_done_max
        done_mask = ~done_mask
        return torch.tensor(done_mask, device=dones.device, dtype=dones.dtype)
