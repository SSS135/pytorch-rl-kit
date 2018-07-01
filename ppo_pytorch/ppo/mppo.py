import copy
import pprint
from collections import namedtuple, OrderedDict
from functools import partial
from itertools import chain

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
from optfn.spectral_norm import spectral_norm, SpectralNorm
from optfn.gadam import GAdam
from collections import deque
from ..models.utils import weights_init
from optfn.grad_running_norm import GradRunningNorm
from optfn.drrelu import DRReLU
from optfn.skip_connections import ResidualBlock


def spectral_init(module, gain=1, n_power_iterations=1):
    nn.init.kaiming_uniform_(module.weight, gain)
    if module.bias is not None:
        module.bias.data.zero_()
    return module # spectral_norm(module, n_power_iterations=n_power_iterations, auto_update_u=False)


def update_spectral_norm(net: nn.Module):
    for module in net.modules():
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, SpectralNorm):
                hook.update_u(module)


def set_lr_scale(optim: torch.optim.Optimizer, scale):
    for group in optim.param_groups:
        group['lr'] = scale * optim.defaults['lr']


def l1_quantile_loss(output, target, tau, reduce=True):
    u = target - output
    loss = (tau - (u.detach() <= 0).float()).mul_(2).mul_(u)

    return loss.mean() if reduce else loss


def huber_quantile_loss(output, target, tau, k=0.05, reduce=True):
    u = target - output
    loss = (tau - (u.detach() <= 0).float()).mul_(2 * u.detach().abs().clamp(max=k).div_(k)).mul_(u)
    return loss.mean() if reduce else loss


class GanG(nn.Module):
    def __init__(self, hidden_code_size, action_pd, hidden_size=512):
        super().__init__()
        self.hidden_code_size = hidden_code_size
        self.action_pd = action_pd
        self.hidden_size = hidden_size
        # self.action_embedding = spectral_init(nn.Linear(action_pd.input_vector_len, hidden_size))
        # self.cur_code_embedding = spectral_init(nn.Linear(hidden_code_size, hidden_size))
        # self.cur_state_embedding = spectral_init(nn.Linear(state_size, hidden_size))
        # self.memory_embedding = spectral_init(nn.Linear(hidden_size, hidden_size))
        # self.memory_out = nn.Linear(hidden_size, hidden_size * 2)
        # self.code_out = nn.Linear(hidden_size, hidden_code_size * 3 + 2)
        input_size = action_pd.input_vector_len + hidden_code_size * 2 + hidden_size + 2
        output_size = hidden_size * 3 + hidden_code_size * 3 + 2
        self.model = nn.Sequential(
            spectral_init(nn.Linear(input_size, hidden_size)),
            nn.LeakyReLU(0.1, True),
            nn.GroupNorm(4, hidden_size),
            spectral_init(nn.Linear(hidden_size, hidden_size)),
            nn.GroupNorm(4, hidden_size),
            nn.LeakyReLU(0.1, True),
            spectral_init(nn.Linear(hidden_size, output_size))
        )

    def forward(self, cur_code, actions, memory, tau):
        action_inputs = self.action_pd.to_inputs(actions)
        # embeddings = [
        #     self.action_embedding(action_inputs),
        #     self.cur_code_embedding(cur_code),
        #     self.memory_embedding(memory),
        #     self.cur_state_embedding(cur_states) if cur_states is not None else torch.zeros_like(memory),
        # ]
        # embeddings = torch.cat(embeddings, -1)
        # embeddings = embeddings.max(-1)[0]
        embeddings = torch.cat([action_inputs, cur_code, memory, tau * 2 - 1], -1)

        next_code, code_f_gate, code_i_gate, rewards, dones, next_memory, memory_f_gate, memory_i_gate = \
            self.model(embeddings).split(3 * [self.hidden_code_size] + [1, 1] + 3 * [self.hidden_size], dim=-1)
        code_f_gate, code_i_gate, memory_f_gate, memory_i_gate = \
            [x.sigmoid() for x in (code_f_gate, code_i_gate, memory_f_gate, memory_i_gate)]
        next_code = code_f_gate * cur_code + code_i_gate * next_code
        next_memory = memory_f_gate * memory + memory_i_gate * next_memory
        dones, rewards = dones.squeeze(-1), rewards.squeeze(-1)
        # dones_mask = (dones > 0).detach()
        # next_code[dones_mask] = cur_states[dones_mask]
        # next_memory[dones_mask] = 0
        return next_code, rewards, dones, next_memory


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
                 replay_buffer_size=16 * 1024,
                 # world_disc_optim_factory=partial(GAdam, lr=4e-4, betas=(0.5, 0.99), weight_decay=1e-4, eps=1e-6),
                 world_gen_optim_factory=partial(GAdam, lr=6e-4, betas=(0.9, 0.9), amsgrad=True, amaxgrad=True,
                                                 amsgrad_decay=0.1, weight_decay=1e-5),
                 # denoiser_optim_factory=partial(GAdam, lr=4e-4, betas=(0.5, 0.99), weight_decay=1e-4, eps=1e-6),
                 world_train_iters=32,
                 world_train_rollouts=32,
                 world_train_horizon=8,
                 began_gamma=0.5,
                 **kwargs):
        super().__init__(*args, **kwargs)
        # assert world_batch_size % world_train_horizon == 0 and \
        #        (world_train_rollouts * world_train_horizon) % world_batch_size == 0
        assert replay_buffer_size >= world_train_iters * world_train_rollouts * (world_train_horizon + 1)

        self.density_buffer_size = density_buffer_size
        self.replay_buffer_size = replay_buffer_size
        # self.world_disc_optim_factory = world_disc_optim_factory
        self.world_gen_optim_factory = world_gen_optim_factory
        self.world_train_iters = world_train_iters
        self.world_train_rollouts = world_train_rollouts
        self.world_train_horizon = world_train_horizon
        self.began_gamma = began_gamma

        self.world_gen = GanG(self.model.hidden_code_size, self.model.pd)
        self.memory_init_model = copy.deepcopy(self.model)
        # self.world_disc = GanD(self.model.hidden_code_size, self.observation_space.shape[0], self.model.pd)
        # self.world_gen_init = GanMemoryInit(self.observation_space.shape[0])
        # self.world_disc_init = GanMemoryInit(self.observation_space.shape[0])
        # self.disc_denoiser = DenoisingAutoencoder()
        self.world_gen_optim = world_gen_optim_factory(
            chain(self.world_gen.parameters(), self.memory_init_model.parameters()))
        # self.world_disc_optim = world_disc_optim_factory(self.world_disc.parameters())
        # self.denoiser_optim = denoiser_optim_factory(self.disc_denoiser.parameters())
        # self.density_buffer = deque(maxlen=density_buffer_size)
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        # self.initial_world_training_done = True

    def _ppo_update(self, data):
        self._update_replay_buffer(data)
        min_buf_size = self.world_train_iters * self.world_train_rollouts * (self.world_train_horizon + 1)
        if len(self.replay_buffer) >= min_buf_size or len(self.replay_buffer) == self.replay_buffer_size:
            # if self.initial_world_training_done:
            self._train_world()
            # else:
            #     # for _ in range(50):
            #     #     self._train_world()
            #     self.initial_world_training_done = True
            #     # for _ in range(10):
            #     #     self._train_world()
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
        # self.world_disc = self.world_disc.to(self.device_train).train()
        # self.world_gen_init = self.world_gen_init.to(self.device_train).train()
        # self.world_disc_init = self.world_disc_init.to(self.device_train).train()
        # self.disc_denoiser = self.disc_denoiser.to(self.device_train).train()
        self.model = self.model.to(self.device_train).train()
        self.memory_init_model = self.memory_init_model.to(self.device_train).train()

        horizon = self.world_train_horizon # np.random.randint(2, 8)
        rollouts = self.world_train_rollouts # 512 // horizon

        # (H, B, ...)
        all_states, all_actions, all_rewards, all_dones = self.replay_buffer.sample(
            rollouts * self.world_train_iters, horizon + 1)

        all_dones = all_dones.astype(np.float32) * 2 - 1
        # all_dones += 0.02 * np.random.randn(*all_dones.shape)
        # all_rewards += 0.02 * np.random.randn(*all_rewards.shape)

        data = [torch.from_numpy(x) for x in (all_states, all_actions, all_rewards, all_dones)]
        if self.device_train.type == 'cuda':
            data = [x.pin_memory() for x in data]

        for train_iter in range(self.world_train_iters):
            slc = (slice(None), slice(train_iter * rollouts, (train_iter + 1) * rollouts))
            # (H, B, ...)
            states, actions, rewards, dones = [x[slc].to(self.device_train) for x in data]

            update_spectral_norm(self.world_gen)
            # update_spectral_norm(self.world_disc)
            # update_spectral_norm(self.disc_denoiser)

            # disc real
            hidden_codes = self.model(states.view(-1, *states.shape[2:]), only_hidden_code_output=True).hidden_code
            hidden_codes = hidden_codes.view(*states.shape[:2], *hidden_codes.shape[1:])
            # hidden_codes += 0.05 * torch.randn_like(hidden_codes)

            # quentile
            with torch.enable_grad():
                all_gen_next_hc = []
                all_gen_rewards = []
                all_gen_dones = []
                tau = hidden_codes.new_empty((hidden_codes.shape[0] - 1, hidden_codes.shape[1], hidden_codes.shape[2] + 2)).uniform_()
                gen_memory = self.memory_init_model(states[0], only_hidden_code_output=True).hidden_code
                for i in range(horizon):
                    gen_next_code, gen_rewards, gen_dones, gen_memory = self.world_gen(
                        hidden_codes[i] if i == 0 else all_gen_next_hc[-1], actions[i], gen_memory, tau[i])
                    all_gen_next_hc.append(gen_next_code)
                    all_gen_rewards.append(gen_rewards)
                    all_gen_dones.append(gen_dones)
                all_gen_next_hc = torch.stack(all_gen_next_hc, 0)
                all_gen_rewards = torch.stack(all_gen_rewards, 0)
                all_gen_dones = torch.stack(all_gen_dones, 0)
                # print(all_gen_hidden_codes.shape, hidden_codes.shape, all_gen_rewards.shape, rewards.shape, all_gen_dones.shape, dones.shape)
                gen_q_loss = huber_quantile_loss(all_gen_next_hc, hidden_codes[1:], tau[..., :-2]) + \
                             0.2 * huber_quantile_loss(all_gen_rewards, rewards[:-1], tau[..., -2]) + \
                             0.2 * huber_quantile_loss(all_gen_dones, dones[:-1], tau[..., -1])

                # gen_head = self.model(all_gen_hidden_codes.view(-1, all_gen_hidden_codes.shape[-1]), hidden_code_input=True)
                # real_head = self.model(hidden_codes[1:].view(-1, hidden_codes.shape[-1]), hidden_code_input=True)
                # head_mse_loss = (gen_head.probs - real_head.probs.detach()).pow(2).mean(-1) + \
                #                 (gen_head.state_values - real_head.state_values.detach()).pow(2).mean(-1)

                # gen_mse_loss = (all_gen_hidden_codes - hidden_codes[1:]).pow(2).mean(-1)
                               # (all_gen_rewards - rewards[:-1]).pow(2) + \
                               # (all_gen_dones - dones[:-1]).pow(2)

                # dones_shift = dones[:-1].clone()
                # dones_shift[1:] = dones[:-2]
                # dones_shift[0] = 0
                # gen_q_loss = gen_q_loss.mul(1 - dones_shift).mean()

                gen_q_loss = gen_q_loss.mean() #+ head_mse_loss.mean()
            gen_q_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.world_gen.parameters(), 20)
            self.world_gen_optim.step()
            self.world_gen_optim.zero_grad()

        if self._do_log:
            self.log_gen_errors('1-step', hidden_codes[0], hidden_codes[1], all_gen_next_hc[0],
                                dones[0], all_gen_dones[0], rewards[0], all_gen_rewards[0], actions[1])
            l = len(all_gen_rewards) - 2
            self.log_gen_errors(f'full-step', hidden_codes[l], hidden_codes[l + 1], all_gen_next_hc[l],
                                dones[l], all_gen_dones[l], rewards[l], all_gen_rewards[l], actions[l + 1])

            self.logger.add_scalar('gen loss', gen_q_loss, self.frame)

    def log_gen_errors(self, tag, real_cur_hidden, real_next_hidden, gen_next_hidden,
                       real_dones, gen_dones, real_rewards, gen_rewards, actions):
        # dones_mask = ((real_dones < 0) & (gen_dones < 0)).float()
        head_real = self.model(real_next_hidden, hidden_code_input=True)
        head_gen = self.model(gen_next_hidden, hidden_code_input=True)
        real_values, real_probs = head_real.state_values, head_real.probs
        gen_values, gen_probs = head_gen.state_values, head_gen.probs
        rmse = lambda a, b: (a - b).abs().mean().item()
        state_norm_rmse = rmse(gen_next_hidden, real_next_hidden) / max(1e-10, rmse(real_cur_hidden, real_next_hidden))
        self.logger.add_scalar(f'gen {tag} raw prob err', rmse(real_probs, gen_probs), self.frame)
        self.logger.add_scalar(f'gen {tag} kl err', self.model.pd.kl(real_probs, gen_probs).mean(), self.frame)
        self.logger.add_scalar(f'gen {tag} abs prob err', rmse(
            self.model.pd.logp(actions, real_probs).exp(),
            self.model.pd.logp(actions, gen_probs).exp()), self.frame)
        self.logger.add_scalar(f'gen {tag} state err', rmse(gen_next_hidden, real_next_hidden), self.frame)
        self.logger.add_scalar(f'gen {tag} state norm err', state_norm_rmse, self.frame)
        self.logger.add_scalar(f'gen {tag} reward err', rmse(gen_rewards, real_rewards), self.frame)
        self.logger.add_scalar(f'gen {tag} done err', rmse(gen_dones, real_dones), self.frame)
        self.logger.add_scalar(f'gen {tag} value err', rmse(real_values, gen_values), self.frame)
        self.logger.add_scalar(f'gen {tag} value mean offset', (real_values - gen_values).mean(), self.frame)
        self.logger.add_scalar(f'gen {tag} prob mean offset', (real_probs - gen_probs).mean(), self.frame)
        value_rms = real_values.pow(2).mean().sqrt()
        self.logger.add_scalar(f'gen {tag} value norm err', rmse(real_values / value_rms, gen_values / value_rms), self.frame)

    # def get_done_mask(self, dones):
    #     done_mask = (dones > 0).cpu().numpy().copy()
    #     done_max = np.zeros(done_mask.shape[1], dtype=np.uint8)
    #     for i in range(done_mask.shape[0]):
    #         new_done_max = np.maximum(done_max, done_mask[i])
    #         done_mask[i] = new_done_max & (~done_mask[i] | done_max)
    #         done_max = new_done_max
    #     done_mask = ~done_mask
    #     return torch.tensor(done_mask, device=dones.device, dtype=dones.dtype)
