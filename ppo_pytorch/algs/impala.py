import copy
import random
from enum import Enum
from functools import partial
from typing import Optional

import math

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
                 vtrace_max_ratio=1.0,
                 vtrace_kl_limit=0.2,
                 upgo_scale=0.2,
                 grad_clip_norm=None,
                 eps_nu_alpha=(3.0, 0.005), # (0.1, 0.005) for V-MPO
                 init_nu_alpha=(1.0, 5.0),
                 kl_limit=0.01,
                 replay_end_sampling_factor=0.1,
                 eval_model_update_interval=5,
                 train_horizon=None,
                 loss_type='impala',
                 eval_model_blend=0.2,
                 smooth_model_blend=False,
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
        self.kl_limit = kl_limit
        self.train_horizon = self.horizon if train_horizon is None else train_horizon
        self.eval_model_blend = eval_model_blend
        self.smooth_model_blend = smooth_model_blend
        # self.batch_size = self.train_horizon * (1 + self.replay_ratio)

        # print(self.horizon // self.train_horizon * self.num_actors, 'x', self.batch_size, 'batches per update')

        assert isinstance(loss_type, str) or isinstance(loss_type, list) or isinstance(loss_type, tuple)
        self.loss_type = (loss_type,) if isinstance(loss_type, str) else loss_type
        self.loss_type = [LossType[c] for c in self.loss_type]
        assert len(set(self.loss_type) - set(c for c in LossType)) == 0

        init_nu_alpha = [max(1e-6, (x + 1) ** 0.5 - 1) for x in init_nu_alpha]
        self.nu_data = torch.scalar_tensor(init_nu_alpha[0], requires_grad=True)
        self.alpha_data = torch.scalar_tensor(init_nu_alpha[1], requires_grad=True)

        # DataLoader limitation
        assert self.batch_size % self.train_horizon == 0 and self.horizon % self.train_horizon == 0, (self.batch_size, self.horizon)

        del self._steps_processor

        self._replay_buffer = ReplayBuffer(replay_buf_size)
        self._prev_data = None
        self._eval_steps = 0
        self._eval_no_copy_updates = 0
        self._adv_norm = RunningNorm(mean_norm=False)
        self._target_model = self.model_factory(self.observation_space, self.action_space).to(self.device_train)
        self._train_future: Optional[Future] = None
        self._data_future: Optional[Future] = None
        self._executor = ThreadPoolExecutor(max_workers=1)

        # self.model_switch_interval = 8
        # self._last_switch_step = 0
        self._discounts = [0.99]
        self._train_models = [self.model_factory(self.observation_space, self.action_space).to(self.device_train)
                              for _ in self._discounts]
        self._eval_models = [copy.deepcopy(m).to(self.device_eval) for m in self._train_models]
        self._optimizers = [self.optimizer_factory(m.parameters()) for m in self._train_models]
        self._lr_schedulers = [self.lr_scheduler_factory(opt) if self.lr_scheduler_factory is not None else None
                               for opt in self._optimizers]
        self._train_model_index = 0
        self._eval_model_index = 0

        for opt in self._optimizers:
            opt.add_param_group(dict(params=[self.nu_data, self.alpha_data]))

        self._switch_model()

    @property
    def nu(self):
        return (self.nu_data + 1.0) ** 2 - 1.0

    @property
    def alpha(self):
        return (self.alpha_data + 1.0) ** 2 - 1.0

    def _step(self, rewards, dones, states) -> torch.Tensor:
        with torch.no_grad():
            # run network
            ac_out = self._eval_model(states.to(self.device_eval))
            ac_out.state_values = ac_out.state_values.squeeze(-1)
            actions = self._eval_model.heads.logits.pd.sample(ac_out.logits).cpu()

            self._eval_steps += 1

            # if self._eval_steps > self._last_switch_step + self.model_switch_interval:
            #     self._last_switch_step = self._eval_steps
            #     index = random.randrange(len(self._lrs_discounts))
            #     self._eval_model = self._eval_models[index]

            if not self.disable_training:
                if self._prev_data is not None and rewards is not None:
                    self._replay_buffer.push(rewards=rewards, dones=dones, **self._prev_data)

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

        self._switch_model()

        self._train_future = self._executor.submit(self._train_async, data)

    def _train_async(self, data):
        with torch.no_grad():
            self._check_log()
            # self._log_training_data(data)
            self._impala_update(data)
            self._model_saver.check_save_model(self._train_model, self.frame_train)
            self._scheduler_step()

    def _switch_model(self):
        self._train_model_index = self._eval_model_index
        self._eval_model_index = random.randrange(len(self._discounts))
        self._train_model = self._train_models[self._train_model_index]
        self._optimizer = self._optimizers[self._train_model_index]
        self.reward_discount = self._discounts[self._train_model_index]
        self._lr_scheduler = self._lr_schedulers[self._train_model_index]
        self._eval_model = self._eval_models[self._eval_model_index]

    # def _switch_model(self):
    #     self._train_model_index = random.randrange(len(self._lrs_discounts))
    #     self._train_model = self._train_models[self._train_model_index]
    #     self._optimizer = self._optimizers[self._train_model_index]
    #     _, self.reward_discount = self._lrs_discounts[self._train_model_index]
    #     self._lr_scheduler = self._lr_schedulers[self._train_model_index]

    def _create_data(self):
        # rand_actors = self.batch_size * self.off_policy_batches // self.horizon
        # rand_samples = self._replay_buffer.sample(rand_actors, self.horizon + 1)
        # return AttrDict(rand_samples)

        def cat_replay(last, rand):
            # return torch.cat([last, rand], 1)
            # (H, B, *) + (H, B * replay, *) = (H, B, 1, *) + (H, B, replay, *) =
            # = (H, B, replay + 1, *) = (H, B * (replay + 1), *)
            H, B, *_ = last.shape
            all = torch.cat([
                last.unsqueeze(2),
                rand.reshape(H, B, self.replay_ratio, *last.shape[2:])], 2)
            return all.reshape(H, B * (self.replay_ratio + 1), *last.shape[2:])

        h_reduce = self.horizon // self.train_horizon

        def fix_on_policy_horizon(v):
            return v.reshape(h_reduce, self.train_horizon, *v.shape[1:])\
                .transpose(0, 1)\
                .reshape(self.train_horizon, h_reduce * v.shape[1], *v.shape[2:])

        # (H, B, *)
        last_samples = self._replay_buffer.get_last_samples(self.horizon)
        last_samples = {k: fix_on_policy_horizon(v) for k, v in last_samples.items()}
        if self.replay_ratio != 0 and len(self._replay_buffer) >= \
                max(self.horizon * self.num_actors * max(1, self.replay_ratio), self.min_replay_size):
            num_rollouts = self.num_actors * self.replay_ratio * h_reduce
            rand_samples = self._replay_buffer.sample(num_rollouts, self.train_horizon, self.replay_end_sampling_factor)
            return AttrDict({k: cat_replay(last, rand)
                             for (k, rand), last in zip(rand_samples.items(), last_samples.values())})
        else:
            return AttrDict(last_samples)

    def _impala_update(self, data: AttrDict):
        eval_model = self._eval_models[self._train_model_index]
        self._target_model.load_state_dict(eval_model.state_dict())

        if self.use_pop_art:
            self._train_model.heads.state_values.normalize(*self._pop_art.statistics)

        num_samples = data.states.shape[0] * data.states.shape[1]
        num_rollouts = data.states.shape[1]

        data = AttrDict(states=data.states, logits_old=data.logits,
                        actions=data.actions, rewards=data.rewards, dones=data.dones)

        num_batches = max(1, num_samples // self.batch_size)
        rand_idx = torch.arange(num_rollouts, device=self.device_train).chunk(num_batches)
        assert len(rand_idx) == num_batches

        old_model = {k: v.clone() for k, v in self._train_model.state_dict().items()}
        kls_policy = []
        kls_replay = []
        value_target_list = []

        with DataLoader(data, rand_idx, self.device_train, 4, dim=1) as data_loader:
            for batch_index in range(num_batches):
                # prepare batch data
                batch = AttrDict(data_loader.get_next_batch())
                loss = self._impala_step(batch, self._do_log and batch_index == num_batches - 1)
                kls_policy.append(batch.kl_policy.mean().item())
                kls_replay.append(batch.kl.mean().item())
                value_target_list.append(batch.value_targets.detach())

        kl_policy = np.mean(kls_policy)
        kl_replay = np.mean(kls_replay)

        if self._do_log:
            self.logger.add_scalar('learning rate', self._learning_rate, self.frame_train)
            self.logger.add_scalar('clip mult', self._clip_mult, self.frame_train)
            if loss is not None:
                self.logger.add_scalar('total loss', loss, self.frame_train)
            self.logger.add_scalar('kl', kl_policy, self.frame_train)
            self.logger.add_scalar('kl_replay', kl_replay, self.frame_train)
            self.logger.add_scalar('kl scale', self.kl_scale, self.frame_train)
            self.logger.add_scalar('model abs diff', model_diff(old_model, self._train_model), self.frame_train)
            self.logger.add_scalar('model max diff', model_diff(old_model, self._train_model, True), self.frame_train)
            self.logger.add_scalar('nu', self.nu, self.frame_train)
            self.logger.add_scalar('alpha', self.alpha, self.frame_train)

        if self.use_pop_art:
            pa_mean, pa_std = self._pop_art.statistics
            if len(value_target_list) != 0:
                value_targets = torch.cat(value_target_list, 0) * pa_std + pa_mean
            self._train_model.heads.state_values.unnormalize(pa_mean, pa_std)
            if len(value_target_list) != 0:
                self._pop_art.update_statistics(value_targets)
            if self._do_log:
                self.logger.add_scalar('pop art mean', pa_mean, self.frame_train)
                self.logger.add_scalar('pop art std', pa_std, self.frame_train)

        # self._adjust_kl_scale(kl)
        # NoisyLinear.randomize_network(self._train_model)

        # self._copy_parameters(self._train_model, eval_modell)
        if self.smooth_model_blend:
            blend_models(self._train_model, eval_model, self.eval_model_blend)
        else:
            self._eval_no_copy_updates += 1
            if self._eval_no_copy_updates >= self.eval_model_update_interval:
                self._eval_no_copy_updates = 0
                self._copy_parameters(self._train_model, eval_model)

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
            loss = self._get_impala_loss(batch, do_log)
            if loss is None:
                return None
            loss = loss.mean()

        # optimize
        loss.backward()
        if self.grad_clip_norm is not None:
            clip_grad_norm_(self._train_model.parameters(), self.grad_clip_norm)
        self._optimizer.step()
        self._optimizer.zero_grad()

        self.nu_data.clamp_(min=math.sqrt(0.1 + 1) - 1)
        self.alpha_data.clamp_(min=math.sqrt(0.01 + 1) - 1)

        return loss

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
        data.logp_old = pd.logp(data.actions, data.logits_old).sum(-1)
        data.logp_policy = pd.logp(data.actions, data.logits_policy).sum(-1)
        data.logp = pd.logp(data.actions, data.logits).sum(-1)
        data.probs_ratio = (data.logp.detach() - data.logp_old).exp()
        data.kl = pd.kl(data.logits, data.logits_old).sum(-1)

        with torch.no_grad():
            self._process_rewards(data)
        data.state_values = data.state_values[:-1]

        for k, v in data.items():
            data[k] = v.flatten(end_dim=1)

        eps_nu, eps_alpha = self.eps_nu_alpha
        # eps_alpha *= self._clip_mult
        data.kl_policy = kl_policy = pd.kl(data.logits, data.logits_policy).sum(-1)
        if LossType.v_mpo in self.loss_type:
            losses = v_mpo_loss(
                kl_policy, data.logp, data.advantages, data.advantages_upgo, data.vtrace_p,
                self.nu, self.alpha, eps_nu, eps_alpha)
            if losses is None:
                return None
            loss_policy, loss_nu, loss_alpha = losses
            loss_policy = loss_policy + loss_nu + loss_alpha
        elif LossType.impala in self.loss_type:
            losses = scaled_impala_loss(
                kl_policy, data.kl, data.logp, data.advantages, data.advantages_upgo, data.vtrace_p,
                self.nu, self.alpha, eps_nu, eps_alpha, self.kl_limit)
            if losses is None:
                return None
            loss_policy, loss_nu, loss_alpha, kurtosis = losses
            loss_policy = loss_policy + loss_nu + loss_alpha

            if do_log:
                self.logger.add_scalar('kurtosis' + tag, kurtosis, self.frame_train)
                self.logger.add_scalar('loss_nu' + tag, loss_nu, self.frame_train)
                self.logger.add_scalar('loss_alpha' + tag, loss_alpha, self.frame_train)

            # loss_alpha = self.alpha * (eps_alpha - kl_policy.detach()) + self.alpha.detach() * kl_policy
            # loss_policy = -data.logp * data.advantages
            # loss_policy = loss_policy.mean() + loss_alpha.mean()

        # adv_u = data.advantages.unsqueeze(-1)
        entropy = pd.entropy(data.logits).sum(-1)
        loss_ent = self.entropy_loss_scale * -entropy
        # loss_policy = -data.logp * adv_u
        loss_value = self.value_loss_scale * barron_loss(data.state_values, data.value_targets, *self.barron_alpha_c, reduce=False)
        # loss_kl = self.kl_scale * kl

        assert loss_value.shape == data.state_values.shape, (loss_value.shape, data.state_values.shape)
        # assert loss_ent.shape == loss_policy.shape, (loss_ent.shape, loss_policy.shape)
        # assert loss_policy.shape == loss_kl.shape, (loss_policy.shape, loss_kl.shape)
        # assert loss_policy.shape == loss_value.shape, (loss_policy.shape, loss_value.shape)
        # assert loss_nu.shape == (), loss_nu.shape
        # assert loss_alpha.shape == (*loss_policy.shape, data.kl.shape[-1]), (loss_alpha.shape, loss_policy.shape)

        loss_ent = loss_ent.mean()
        # loss_policy = loss_policy.sum()
        # loss_nu = loss_nu.mean()
        # loss_alpha = loss_alpha.mean()
        # loss_kl = loss_kl.mean()
        loss_value = loss_value.mean()

        # sum all losses
        total_loss = loss_policy + loss_value + loss_ent #+ loss_kl
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

        return total_loss

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

        advantages_upgo *= self.upgo_scale
        if self.use_pop_art:
            value_targets = (value_targets - pa_mean) / pa_std
            # if LossType.impala is self.loss_type:
            #     advantages /= pa_std
            #     advantages_upgo /= pa_std

        # if LossType.impala is self.loss_type:
        # advantages = self._adv_norm(advantages)
        # advantages_upgo = self._adv_norm(advantages_upgo, update_stats=False)

        data.vtrace_p, data.advantages_upgo = p, advantages_upgo
        data.value_targets, data.advantages, data.rewards = value_targets, advantages, norm_rewards

    def drop_collected_steps(self):
        self._prev_data = None