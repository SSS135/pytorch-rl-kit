import random
from enum import Enum
from functools import partial
from typing import Optional

import math
from copy import deepcopy

import numpy as np
import torch
from ..algs.utils import blend_models
from torch.optim.rmsprop import RMSprop
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from rl_exp.noisy_linear import NoisyLinear
from torch import nn

from .ppo import PPO
from .replay_buffer import ReplayBuffer
from ..common.attr_dict import AttrDict
from ..common.barron_loss import barron_loss_derivative, barron_loss
from ..common.data_loader import DataLoader
from ..common.gae import calc_vtrace, calc_advantages, calc_value_targets
from ..actors.utils import model_diff
from .utils import v_mpo_loss, RunningNorm, scaled_impala_loss
from asyncio import Future
from concurrent.futures import ThreadPoolExecutor


class LossType(Enum):
    v_mpo = 'v_mpo'
    impala = 'impala'


class IMPALA(PPO):
    def __init__(self, *args,
                 replay_buf_size=256 * 1024,
                 replay_ratio=7,
                 min_replay_size=10000,
                 vtrace_max_ratio=2.0,
                 vtrace_kl_limit=0.2,
                 upgo_scale=0.2,
                 grad_clip_norm=None,
                 eps_nu_alpha=(1.7639, 0.005), # (0.1, 0.005) for V-MPO
                 init_nu_alpha=(0.5, 5.0), # (1.0, 5.0) for V-MPO
                 replay_end_sampling_factor=0.1,
                 eval_model_update_interval=1,
                 train_horizon=None,
                 loss_type='impala',
                 use_pop_art=True,
                 **kwargs):
        super().__init__(*args, grad_clip_norm=grad_clip_norm, use_pop_art=use_pop_art, **kwargs)
        self.replay_buf_size = replay_buf_size
        self.replay_ratio = replay_ratio
        self.vtrace_max_ratio = vtrace_max_ratio
        self.vtrace_kl_limit = vtrace_kl_limit
        self.upgo_scale = upgo_scale
        self.min_replay_size = min_replay_size
        self.replay_end_sampling_factor = replay_end_sampling_factor
        self.eps_nu_alpha = eps_nu_alpha
        self.eval_model_update_interval = eval_model_update_interval
        self.train_horizon = self.horizon if train_horizon is None else train_horizon

        assert isinstance(loss_type, str) or isinstance(loss_type, list) or isinstance(loss_type, tuple)
        self.loss_type = (loss_type,) if isinstance(loss_type, str) else loss_type
        self.loss_type = [LossType[c] for c in self.loss_type]
        assert len(set(self.loss_type) - set(c for c in LossType)) == 0

        self.nu_data = torch.scalar_tensor(init_nu_alpha[0] ** 0.5, requires_grad=True)
        self.alpha_data = torch.scalar_tensor(init_nu_alpha[1] ** 0.5, requires_grad=True)
        self._optimizer.add_param_group(dict(params=[self.nu_data, self.alpha_data]))

        # DataLoader limitation
        assert self.batch_size % self.train_horizon == 0 and self.horizon % self.train_horizon == 0, (self.batch_size, self.horizon)

        del self._steps_processor

        self._replay_buffer = ReplayBuffer(replay_buf_size)
        self._prev_data = None
        self._eval_steps = 0
        self._eval_no_copy_updates = 0
        self._adv_norm = RunningNorm(False)
        self._target_model = None
        self._train_future: Optional[Future] = None
        self._data_future: Optional[Future] = None
        self._executor = ThreadPoolExecutor(max_workers=2)

    @property
    def nu(self):
        sum_sq = (self.nu_data + 1.0) ** 2
        return sum_sq - (sum_sq - self.nu_data ** 2).detach()

    @property
    def alpha(self):
        sum_sq = (self.alpha_data + 1.0) ** 2
        return sum_sq - (sum_sq - self.alpha_data ** 2).detach()

    def _step(self, rewards, dones, states) -> torch.Tensor:
        with torch.no_grad():
            # run network
            ac_out = self._take_step(states.to(self.device_eval), dones)
            ac_out.state_values = ac_out.state_values.squeeze(-1)
            actions = self._train_model.heads.logits.pd.sample(ac_out.logits).cpu()

            if not self.disable_training:
                if self._prev_data is not None and rewards is not None:
                    self._replay_buffer.push(rewards=rewards, dones=dones, **self._prev_data)

                self._eval_steps += 1
                self._prev_data = AttrDict(**ac_out, states=states, actions=actions)

                if self._eval_steps >= self.horizon + 2:
                    self._eval_steps = 0
                    self._train()

            return actions

    def _train(self):
        self.step_train = self.step_eval

        data = self._create_data()

        # self._train_async(data)
        # if self._data_future is not None:
        #     data = self._data_future.result()
        # else:
        #     data = self._create_data()
        # self._data_future = self._executor.submit(self._create_data)

        if self._train_future is not None:
            self._train_future.result()
        self._train_future = self._executor.submit(self._train_async, data)

    def _train_async(self, data):
        with torch.no_grad():
            self._check_log()
            # self._log_training_data(data)
            self._impala_update(data)
            self._model_saver.check_save_model(self._train_model, self.frame_train)
            self._scheduler_step()

    def _create_data(self):
        # rand_actors = self.batch_size * self.off_policy_batches // self.horizon
        # rand_samples = self._replay_buffer.sample(rand_actors, self.horizon + 1)
        # return AttrDict(rand_samples)

        h_reduce = self.horizon // self.train_horizon
        last_samples = self._replay_buffer.get_last_samples(self.horizon)
        last_samples = {k: v.reshape(v.shape[0] // h_reduce, v.shape[1] * h_reduce, *v.shape[2:]) for k, v in last_samples.items()}
        if self.replay_ratio != 0 and len(self._replay_buffer) >= \
                max(self.horizon * self.num_actors * max(1, self.replay_ratio), self.min_replay_size):
            num_rollouts = self.num_actors * self.replay_ratio * h_reduce
            rand_samples = self._replay_buffer.sample(num_rollouts, self.train_horizon, self.replay_end_sampling_factor)
            return AttrDict({k: torch.cat([v1, v2], 1) for (k, v1), v2 in zip(rand_samples.items(), last_samples.values())})
        else:
            return AttrDict(last_samples)

    def _impala_update(self, data: AttrDict):
        self._target_model = self._eval_model.to(self.device_train)

        if self.use_pop_art:
            self._train_model.heads.state_values.normalize(*self._pop_art.statistics)

        num_samples = data.states.shape[0] * data.states.shape[1]
        num_rollouts = data.states.shape[1]

        data = AttrDict(states=data.states, logits_old=data.logits,
                        actions=data.actions, rewards=data.rewards, dones=data.dones)

        num_batches = num_samples // self.batch_size
        rand_idx = torch.arange(num_rollouts, device=self.device_train).chunk(num_batches)
        assert len(rand_idx) == num_batches

        old_model = deepcopy(self._train_model)
        kl_list = []
        value_target_list = []

        with DataLoader(data, rand_idx, self.device_train, 4, dim=1) as data_loader:
            for batch_index in range(num_batches):
                # prepare batch data
                batch = AttrDict(data_loader.get_next_batch())
                loss, kl = self._impala_step(batch, self._do_log and batch_index == num_batches - 1)
                kl_list.append(kl)
                value_target_list.append(batch.value_targets.detach())

        kl = np.mean(kl_list)

        if self._do_log:
            self.logger.add_scalar('learning rate', self._learning_rate, self.frame_train)
            self.logger.add_scalar('clip mult', self._clip_mult, self.frame_train)
            if loss is not None:
                self.logger.add_scalar('total loss', loss, self.frame_train)
            self.logger.add_scalar('kl', kl, self.frame_train)
            self.logger.add_scalar('kl scale', self.kl_scale, self.frame_train)
            self.logger.add_scalar('model abs diff', model_diff(old_model, self._train_model), self.frame_train)
            self.logger.add_scalar('model max diff', model_diff(old_model, self._train_model, True), self.frame_train)
            self.logger.add_scalar('nu', self.nu, self.frame_train)
            self.logger.add_scalar('alpha', self.alpha, self.frame_train)

        if self.use_pop_art:
            pa_mean, pa_std = self._pop_art.statistics
            value_targets = torch.cat(value_target_list, 0) * pa_std + pa_mean
            self._train_model.heads.state_values.unnormalize(pa_mean, pa_std)
            self._pop_art.update_statistics(value_targets)
            if self._do_log:
                self.logger.add_scalar('pop art mean', pa_mean, self.frame_train)
                self.logger.add_scalar('pop art std', pa_std, self.frame_train)

        # self._adjust_kl_scale(kl)
        NoisyLinear.randomize_network(self._train_model)

        # self._copy_parameters(self._train_model, self._eval_model)
        # blend_models(self._train_model, self._eval_model, self.eval_model_blend)
        self._eval_no_copy_updates += 1
        if self._eval_no_copy_updates >= self.eval_model_update_interval:
            self._eval_no_copy_updates = 0
            self._copy_parameters(self._train_model, self._eval_model)

    def _impala_step(self, batch, do_log):
        with torch.enable_grad():
            actor_params = AttrDict()
            if do_log:
                actor_params.logger = self.logger
                actor_params.cur_step = self.frame_train

            actor_out = self._train_model(batch.states.reshape(-1, *batch.states.shape[2:]), **actor_params)
            with torch.no_grad():
                actor_out_policy = self._target_model(batch.states.reshape(-1, *batch.states.shape[2:]))

            batch.logits = actor_out.logits.reshape(*batch.states.shape[:2], *actor_out.logits.shape[1:])
            batch.logits_policy = actor_out_policy.logits.reshape(*batch.states.shape[:2], *actor_out.logits.shape[1:])
            batch.state_values = actor_out.state_values.reshape(*batch.states.shape[:2])

            for k, v in list(batch.items()):
                batch[k] = v if k == 'states' else v.cpu()

            # get loss
            loss, kl = self._get_impala_loss(batch, do_log)
            loss = loss.mean()

        kl = kl.item()

        # optimize
        loss.backward()
        if self.grad_clip_norm is not None:
            clip_grad_norm_(self._train_model.parameters(), self.grad_clip_norm)
        self._optimizer.step()
        self._optimizer.zero_grad()

        self.nu_data.clamp_(min=1e-8)
        self.alpha_data.clamp_(min=1e-8)

        return loss, kl

    def _get_impala_loss(self, data, do_log=False, pd=None, tag=''):
        """
        Single iteration of PPO algorithm.
        Returns: Total loss and KL divergence.
        """

        if pd is None:
            pd = self._train_model.heads.logits.pd

        state_values = data.state_values
        data.update({k: v[:-1] for k, v in data.items()})
        data.state_values = state_values

        # action probability ratio
        # log probabilities used for better numerical stability
        data.logp_old = pd.logp(data.actions, data.logits_old).sum(-1)
        data.logp_policy = pd.logp(data.actions, data.logits_policy).sum(-1)
        data.logp = pd.logp(data.actions, data.logits).sum(-1)
        data.probs_ratio = (data.logp.detach() - data.logp_old).exp()
        data.kl = pd.kl(data.logits_old, data.logits).sum(-1)

        with torch.no_grad():
            self._process_rewards(data)
        data.state_values = data.state_values[:-1]

        data = AttrDict({k: v.flatten(end_dim=1) for k, v in data.items()})

        eps_nu, eps_alpha = self.eps_nu_alpha
        kl_policy = pd.kl(data.logits_policy, data.logits).sum(-1)
        if LossType.v_mpo in self.loss_type:
            loss_policy, loss_nu, loss_alpha = v_mpo_loss(
                kl_policy, data.logp, data.advantages, data.advantages_upgo, data.vtrace_p,
                self.nu, self.alpha, eps_nu, eps_alpha)
            loss_policy = loss_policy + loss_nu + loss_alpha
        elif LossType.impala in self.loss_type:
            loss_policy, loss_nu, loss_alpha = scaled_impala_loss(
                kl_policy, data.logp, data.advantages, data.advantages_upgo, data.vtrace_p,
                self.nu, self.alpha, eps_nu, eps_alpha)
            loss_policy = loss_policy + loss_nu + loss_alpha

            # loss_alpha = self.alpha * (eps_alpha - kl_policy.detach()) + self.alpha.detach() * kl_policy
            # loss_policy = -data.logp * data.advantages
            # loss_policy = loss_policy.mean() + loss_alpha.mean()

        # adv_u = data.advantages.unsqueeze(-1)
        entropy = pd.entropy(data.logits).sum(-1)
        # loss_ent = self.entropy_loss_scale * -entropy
        # loss_policy = -data.logp * adv_u
        loss_value = self.value_loss_scale * barron_loss(data.state_values, data.value_targets, *self.barron_alpha_c, reduce=False)
        # loss_kl = self.kl_scale * kl

        assert loss_value.shape == data.state_values.shape, (loss_value.shape, data.state_values.shape)
        # assert loss_ent.shape == loss_policy.shape, (loss_ent.shape, loss_policy.shape)
        # assert loss_policy.shape == loss_kl.shape, (loss_policy.shape, loss_kl.shape)
        # assert loss_policy.shape == loss_value.shape, (loss_policy.shape, loss_value.shape)
        # assert loss_nu.shape == (), loss_nu.shape
        # assert loss_alpha.shape == (*loss_policy.shape, data.kl.shape[-1]), (loss_alpha.shape, loss_policy.shape)
        #
        # loss_ent = loss_ent.mean()
        # loss_policy = loss_policy.sum()
        # loss_nu = loss_nu.mean()
        # loss_alpha = loss_alpha.mean()
        # loss_kl = loss_kl.mean()
        loss_value = loss_value.mean()

        # sum all losses
        total_loss = loss_policy + loss_value #+ loss_ent #+ loss_kl
        assert not np.isnan(total_loss.mean().item()) and not np.isinf(total_loss.mean().item()), \
            (loss_policy.mean().item(), loss_value.mean().item())

        with torch.no_grad():
            if do_log:
                self._log_training_data(data)
                ratio = (data.logp - data.logp_policy).exp() - 1
                self.logger.add_scalar('ratio mean' + tag, ratio.mean(), self.frame_train)
                self.logger.add_scalar('ratio abs mean' + tag, ratio.abs().mean(), self.frame_train)
                self.logger.add_scalar('ratio abs max' + tag, ratio.abs().max(), self.frame_train)
                self.logger.add_scalar('success updates', data.vtrace_p.mean(), self.frame_train)
                self.logger.add_scalar('entropy' + tag, entropy.mean(), self.frame_train)
                # self.logger.add_scalar('loss entropy' + tag, loss_ent.mean(), self.frame)
                self.logger.add_scalar('loss state value' + tag, loss_value.mean(), self.frame_train)
                if LossType.v_mpo is self.loss_type:
                    self.logger.add_scalar('loss nu' + tag, loss_nu, self.frame_train)
                    self.logger.add_scalar('loss alpha' + tag, loss_alpha, self.frame_train)
                self.logger.add_histogram('loss value hist' + tag, loss_value, self.frame_train)
                # self.logger.add_histogram('loss ent hist' + tag, loss_ent, self.frame)
                self.logger.add_histogram('ratio hist' + tag, ratio, self.frame_train)

        return total_loss, kl_policy.mean()

    def _process_rewards(self, data, mean_norm=True):
        norm_rewards = self.reward_scale * data.rewards

        if self.use_pop_art:
            pa_mean, pa_std = self._pop_art.statistics

        state_values = data.state_values.detach() * pa_std + pa_mean if self.use_pop_art else data.state_values.detach()
        # calculate value targets and advantages
        value_targets, advantages, advantages_upgo, p = calc_vtrace(
            norm_rewards, state_values,
            data.dones, data.probs_ratio.detach(), data.kl.detach(),
            self.reward_discount, self.vtrace_max_ratio, self.vtrace_kl_limit)

        if self.use_pop_art:
            value_targets = (value_targets - pa_mean) / pa_std
            if LossType.impala is self.loss_type:
                advantages /= pa_std
                advantages_upgo /= pa_std

        if LossType.impala is self.loss_type:
            advantages = self._adv_norm(advantages)
            advantages_upgo = self._adv_norm(advantages_upgo, update_stats=False)

        data.vtrace_p, data.advantages_upgo = p, self.upgo_scale * advantages_upgo
        data.value_targets, data.advantages, data.rewards = value_targets, advantages, norm_rewards

    def drop_collected_steps(self):
        self._prev_data = None