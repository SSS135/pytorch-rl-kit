from functools import partial
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
from optfn.gadam import GAdam
from optfn.spectral_norm import SpectralNorm

from .ppo import PPO


def spectral_init(module, gain=nn.init.calculate_gain('leaky_relu', 0.1), n_power_iterations=1):
    nn.init.orthogonal_(module.weight, gain=gain)
    # if module.bias is not None:
    #     module.bias.data.zero_()
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
    def __init__(self, latent_input_code_size, action_pd, hidden_size=512):
        super().__init__()
        self.latent_input_code_size = latent_input_code_size
        self.action_pd = action_pd
        self.hidden_size = hidden_size
        input_size = action_pd.input_vector_len + latent_input_code_size * 2 + hidden_size + 2 + action_pd.prob_vector_len + 1
        output_size = hidden_size * 3 + latent_input_code_size * 3 + 2
        self.model = nn.Sequential(
            spectral_init(nn.Linear(input_size, hidden_size)),
            nn.LeakyReLU(0.1, True),
            nn.GroupNorm(4, hidden_size),
            spectral_init(nn.Linear(hidden_size, hidden_size, bias=False)),
            nn.GroupNorm(4, hidden_size),
            nn.LeakyReLU(0.1, True),
            spectral_init(nn.Linear(hidden_size, output_size))
        )
        self.memory_init = nn.Sequential(
            spectral_init(nn.Linear(latent_input_code_size, hidden_size, bias=False)),
            nn.GroupNorm(4, hidden_size),
            nn.LeakyReLU(0.1, True),
        )

        split_w = self.model[0].weight.split([
            action_pd.input_vector_len, latent_input_code_size, hidden_size,
            latent_input_code_size, action_pd.prob_vector_len, 3
        ], 1)
        for w in split_w:
            nn.init.orthogonal_(w, gain=nn.init.calculate_gain('leaky_relu', 0.1))

    def forward(self, cur_code, actions, memory, tau, memory_init):
        if memory is None:
            memory = self.memory_init(memory_init)
        action_inputs = self.action_pd.to_inputs(actions)
        embeddings = torch.cat([action_inputs, cur_code, memory, tau * 2 - 1], -1)

        next_code, code_f_gate, code_i_gate, rewards, dones, next_memory, memory_f_gate, memory_i_gate = \
            self.model(embeddings).split(3 * [self.latent_input_code_size] + [1, 1] + 3 * [self.hidden_size], dim=-1)
        code_f_gate, code_i_gate, memory_f_gate, memory_i_gate = \
            [x.sigmoid() for x in (code_f_gate, code_i_gate, memory_f_gate, memory_i_gate)]
        next_code = code_f_gate * cur_code + code_i_gate * next_code
        next_memory = memory_f_gate * memory + memory_i_gate * next_memory
        dones, rewards = dones.squeeze(-1).mul(0.5).add_(0.5), rewards.squeeze(-1)
        return next_code, rewards, dones, next_memory


class ReplayBuffer:
    def __init__(self, per_actor_capacity):
        self.per_actor_capacity = per_actor_capacity
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
            self.states = np.zeros((self.per_actor_capacity, actors, *states.shape[2:]), dtype=states.dtype)
            self.actions = np.zeros((self.per_actor_capacity, actors, *actions.shape[2:]), dtype=actions.dtype)
            self.rewards = np.zeros((self.per_actor_capacity, actors), dtype=np.float32)
            self.dones = np.zeros((self.per_actor_capacity, actors), dtype=np.uint8)

        if self.index + states.shape[0] <= self.per_actor_capacity:
            self._push_unchecked(states, actions, rewards, dones)
        else:
            n = self.per_actor_capacity - self.index
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
            rand_h = np.random.randint(self.per_actor_capacity - horizon)
            src_slice = (slice(rand_h, rand_h + horizon), rand_r)
            dst_slice = (slice(None), ri)
            states[dst_slice] = self.states[src_slice]
            actions[dst_slice] = self.actions[src_slice]
            rewards[dst_slice] = self.rewards[src_slice]
            dones[dst_slice] = self.dones[src_slice]

        return states, actions, rewards, dones

    def __len__(self):
        return (self.per_actor_capacity if self.full_loop else self.index) * \
               (self.states.shape[1] if self.states is not None else 1)


class MPPO(PPO):
    def __init__(self, *args,
                 density_buffer_size=2 * 1024,
                 per_actor_replay_buffer_size=4 * 1024,
                 world_gen_optim_factory=partial(GAdam, lr=1e-3, betas=(0.9, 0.95), amsgrad=True, amaxgrad=True,
                                                 amsgrad_decay=0.05, weight_decay=1e-5),
                 world_train_iters=32,
                 world_train_rollouts=32,
                 world_train_horizon=8,
                 began_gamma=0.5,
                 **kwargs):
        super().__init__(*args, **kwargs)
        # assert world_batch_size % world_train_horizon == 0 and \
        #        (world_train_rollouts * world_train_horizon) % world_batch_size == 0
        assert per_actor_replay_buffer_size * self.num_actors >= world_train_iters * world_train_rollouts * (world_train_horizon + 1)

        self.density_buffer_size = density_buffer_size
        self.per_actor_replay_buffer_size = per_actor_replay_buffer_size
        self.world_gen_optim_factory = world_gen_optim_factory
        self.world_train_iters = world_train_iters
        self.world_train_rollouts = world_train_rollouts
        self.world_train_horizon = world_train_horizon
        self.began_gamma = began_gamma

        self.world_gen = GanG(self.model.hidden_code_size, self.model.pd)
        self.memory_init_model = self.model_factory(self.observation_space, self.action_space)
        for src, dst in zip(self.model.parameters(), self.memory_init_model.parameters()):
            dst.data.copy_(src.data)
        self.world_gen_optim = world_gen_optim_factory(
            chain(self.world_gen.parameters(), self.memory_init_model.parameters()))
        # self.density_buffer = deque(maxlen=density_buffer_size)
        self.replay_buffer = ReplayBuffer(per_actor_replay_buffer_size)

    def _ppo_update(self, data):
        self._update_replay_buffer(data)
        min_buf_size = self.world_train_iters * self.world_train_rollouts * (self.world_train_horizon + 1)
        if len(self.replay_buffer) >= min_buf_size or len(self.replay_buffer) == self.per_actor_replay_buffer_size * self.num_actors:
            self._train_world()
        if self._do_log:
            self._test_world(data)
        return super()._ppo_update(data)

    def _update_replay_buffer(self, data):
        # H x B x *
        states, actions, rewards, dones = [x.view(-1, self.num_actors, *x.shape[1:])
                                           for x in (data.states, data.actions, data.rewards, data.dones)]
        self.replay_buffer.push(states, actions, rewards, dones)

    def _train_world(self):
        # move model to cuda or cpu
        self.world_gen = self.world_gen.to(self.device_train).train()
        self.model = self.model.to(self.device_train).train()
        self.memory_init_model = self.memory_init_model.to(self.device_train).train()

        horizon = self.world_train_horizon # np.random.randint(2, 8)
        rollouts = self.world_train_rollouts # 512 // horizon

        # (H, B, ...)
        all_states, all_actions, all_rewards, all_dones = self.replay_buffer.sample(
            rollouts * self.world_train_iters, horizon + 1)
        all_dones = all_dones.astype(np.float32)

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

            # quentile
            with torch.enable_grad():
                real_head = self.model(states.view(-1, *states.shape[2:]))
                hidden_codes = real_head.hidden_code.detach().view(*states.shape[:2], *real_head.hidden_code.shape[1:])

                all_gen_next_hc, all_gen_rewards, all_gen_dones, tau_hc, tau_prob, tau_done, tau_reward, tau_value = \
                    self.run_generator(hidden_codes, states, actions)
                correct_dones_mask = (all_gen_dones.detach() > 0.5) == (dones[:-1] > 0.5)
                all_gen_dones[correct_dones_mask] = all_gen_dones[correct_dones_mask].clamp(0, 1)
                gen_q_loss = huber_quantile_loss(all_gen_next_hc, hidden_codes[1:], tau_hc) + \
                             huber_quantile_loss(all_gen_rewards, rewards[:-1], tau_reward) + \
                             huber_quantile_loss(all_gen_dones, dones[:-1], tau_done)

                # gen_head = self.model(all_gen_next_hc.view(-1, all_gen_next_hc.shape[-1]), hidden_code_input=True)
                # # real_head = self.model(hidden_codes[1:].view(-1, hidden_codes.shape[-1]), hidden_code_input=True)
                # real_probs = real_head.probs.detach().view(-1, states.shape[1], *real_head.probs.shape[1:])
                # gen_probs = gen_head.probs.view(-1, states.shape[1], *gen_head.probs.shape[1:])
                # real_values = real_head.state_values.detach().view(-1, states.shape[1])
                # gen_values = gen_head.state_values.view(-1, states.shape[1])
                # head_q_loss = huber_quantile_loss(gen_probs, real_probs[1:], tau_prob) + \
                #               huber_quantile_loss(gen_values, real_values[1:], tau_value)

                # gen_mse_loss = (all_gen_hidden_codes - hidden_codes[1:]).pow(2).mean(-1)
                               # (all_gen_rewards - rewards[:-1]).pow(2) + \
                               # (all_gen_dones - dones[:-1]).pow(2)

                dones_shift = dones[:-1].clone()
                dones_shift[1:] = dones[:-2]
                dones_shift[0] = 0
                gen_q_loss = gen_q_loss.mul(1 - dones_shift).mean()

                gen_q_loss = gen_q_loss.mean() #+ 0.2 * head_q_loss.mean() #+ head_mse_loss.mean()
            gen_q_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.world_gen.parameters(), 20)
            self.optimizer.zero_grad()
            self.world_gen_optim.step()
            self.world_gen_optim.zero_grad()

        if self._do_log:
            self.logger.add_scalar('gen loss', gen_q_loss, self.frame)

    def run_generator(self, hidden_codes, states, actions):
        all_gen_next_hc = []
        all_gen_rewards = []
        all_gen_dones = []
        prob_len = self.model.pd.prob_vector_len
        hc_len = hidden_codes.shape[2]
        tau = hidden_codes.new_empty((hidden_codes.shape[0] - 1, hidden_codes.shape[1], hc_len + 2 + prob_len + 1)).uniform_()
        tau_hc, tau_prob, tau_done, tau_reward, tau_value = tau.split([hc_len, prob_len, 1, 1, 1], -1)
        tau_done, tau_reward, tau_value = [x.squeeze(-1) for x in (tau_done, tau_reward, tau_value)]
        init_state = self.memory_init_model(states[0], only_hidden_code_output=True).hidden_code
        gen_memory = None
        for i in range(hidden_codes.shape[0] - 1):
            gen_next_code, gen_rewards, gen_dones, gen_memory = self.world_gen(
                hidden_codes[0] if i == 0 else all_gen_next_hc[-1], actions[i],
                gen_memory, tau[i], init_state if i == 0 else None)
            all_gen_next_hc.append(gen_next_code)
            all_gen_rewards.append(gen_rewards)
            all_gen_dones.append(gen_dones)
        all_gen_next_hc = torch.stack(all_gen_next_hc, 0)
        all_gen_rewards = torch.stack(all_gen_rewards, 0)
        all_gen_dones = torch.stack(all_gen_dones, 0)
        return all_gen_next_hc, all_gen_rewards, all_gen_dones, tau_hc, tau_prob, tau_done, tau_reward, tau_value

    def _test_world(self, data):
        self.world_gen = self.world_gen.to(self.device_train).train()
        self.memory_init_model = self.memory_init_model.to(self.device_train).train()
        self.model = self.model.to(self.device_train).train()

        world_horizon = self.world_train_horizon
        data_horizon = data.states.shape[0] // self.num_actors
        fix_data_horizon = data_horizon - data_horizon % (world_horizon + 1)
        rollouts = fix_data_horizon // (world_horizon + 1) * self.num_actors
        states, actions, rewards, dones = [
            x.to(self.device_train)
                .view(-1, self.num_actors, *x.shape[1:])[:fix_data_horizon]
                .view(rollouts // self.num_actors, world_horizon + 1, self.num_actors, *x.shape[1:])
                .transpose(0, 1).contiguous()
                .view(world_horizon + 1, -1, *x.shape[1:])
            for x in (data.states, data.actions, data.rewards, data.dones)
        ]
        dones = dones.float()

        real_head = self.model(states.view(-1, *states.shape[2:]))
        hidden_codes = real_head.hidden_code.detach().view(*states.shape[:2], *real_head.hidden_code.shape[1:])
        all_gen_next_hc, all_gen_rewards, all_gen_dones, tau_hc, tau_prob, tau_done, tau_reward, tau_value = \
            self.run_generator(hidden_codes, states, actions)
        all_gen_dones = (all_gen_dones > 0.5).float()

        self._log_gen_errors('1-step', hidden_codes[0], hidden_codes[1], all_gen_next_hc[0],
                             dones[0], all_gen_dones[0], rewards[0], all_gen_rewards[0], actions[1])
        l = all_gen_rewards.shape[0] - 1
        self._log_gen_errors(f'full-step', hidden_codes[l], hidden_codes[l + 1], all_gen_next_hc[l],
                             dones[l], all_gen_dones[l], rewards[l], all_gen_rewards[l], actions[l + 1])

    def _log_gen_errors(self, tag, real_cur_hidden, real_next_hidden, gen_next_hidden,
                        real_dones, gen_dones, real_rewards, gen_rewards, actions):
        head_real = self.model(real_next_hidden, hidden_code_input=True)
        head_gen = self.model(gen_next_hidden, hidden_code_input=True)
        real_values, real_probs = head_real.state_values, head_real.probs
        gen_values, gen_probs = head_gen.state_values, head_gen.probs
        rmse = lambda a, b: (a - b).abs().mean().item()
        state_norm_rmse = rmse(gen_next_hidden, real_next_hidden) / max(0.01, rmse(real_cur_hidden, real_next_hidden))
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

