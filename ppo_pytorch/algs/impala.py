import math
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from .ppo import PPO
from .replay_buffer import ReplayBuffer
from ..common.attr_dict import AttrDict
from ..common.barron_loss import barron_loss_derivative, barron_loss
from ..common.data_loader import DataLoader
from ..common.gae import calc_vtrace, calc_advantages
from ..actors.utils import model_diff


class IMPALA(PPO):
    def __init__(self, *args, replay_buf_size=32 * 1024, off_policy_batches=4,
                 vtrace_c_max=1.0, vtrace_p_max=1.0, vtrace_kl_limit=0.3,
                 kl_scale=0.01, **kwargs):
        super().__init__(*args, kl_scale=kl_scale, **kwargs)
        self.replay_buf_size = replay_buf_size
        self.off_policy_batches = off_policy_batches
        self.vtrace_c_max = vtrace_c_max
        self.vtrace_p_max = vtrace_p_max
        self.vtrace_kl_limit = vtrace_kl_limit

        assert self.batch_size % self.horizon == 0

        del self._steps_processor

        self._replay_buffer = ReplayBuffer(replay_buf_size)
        self._prev_data = None
        self._eval_steps = 0
        self._advantage_stats = (0, 0, 0)
        self._advantage_momentum = 0.99

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

                if self._eval_steps >= self.horizon + 1 and len(self._replay_buffer) >= (self.horizon + 1) * self.num_actors:
                    self._eval_steps = 0
                    self._pre_train()
                    self._train()

            return actions

    def _train(self):
        data = self._create_data()
        self._train_async(data)
        # if self._train_future is not None:
        #     self._train_future.result()
        # self._train_future = self._train_executor.submit(self._train_async, data)

    def _train_async(self, data):
        with torch.no_grad():
            # self._log_training_data(data)
            self._impala_update(data)
            self._model_saver.check_save_model(self._train_model, self.frame)

    def _create_data(self):
        # rand_actors = self.batch_size * self.off_policy_batches // self.horizon
        # rand_samples = self._replay_buffer.sample(rand_actors, self.horizon + 1)
        # return AttrDict(rand_samples)

        last_samples = self._replay_buffer.get_last_samples(self.horizon + 1)
        if self.off_policy_batches != 0:
            rand_actors = self.batch_size * self.off_policy_batches // self.horizon
            rand_samples = self._replay_buffer.sample(rand_actors, self.horizon + 1)
            return AttrDict({k: torch.cat([v1, v2], 1) for (k, v1), v2 in zip(rand_samples.items(), last_samples.values())})
        else:
            return AttrDict(last_samples)

    def _impala_update(self, data: AttrDict):
        num_samples = data.states.shape[0] * data.states.shape[1]
        num_rollouts = data.states.shape[1]

        data = AttrDict(states=data.states, logits_old=data.logits, state_values_old=data.state_values,
                        actions=data.actions, rewards=data.rewards, dones=data.dones)

        num_batches = num_samples // self.batch_size
        rand_idx = torch.randperm(num_rollouts, device=self.device_train).chunk(num_batches)

        old_model = deepcopy(self._train_model)
        kl_list = []

        with DataLoader(data, rand_idx, self.device_train, 4, dim=1) as data_loader:
            for batch_index in range(num_batches):
                # prepare batch data
                batch = AttrDict(data_loader.get_next_batch())
                loss, kl = self._impala_step(batch, self._do_log and batch_index == num_batches - 1)
                kl_list.append(kl)

        kl = np.mean(kl_list)

        if self._do_log:
            self.logger.add_scalar('learning rate', self._learning_rate, self.frame)
            self.logger.add_scalar('clip mult', self._clip_mult, self.frame)
            self.logger.add_scalar('total loss', loss, self.frame)
            self.logger.add_scalar('kl', kl, self.frame)
            self.logger.add_scalar('kl scale', self.kl_scale, self.frame)
            self.logger.add_scalar('model abs diff', model_diff(old_model, self._train_model), self.frame)
            self.logger.add_scalar('model max diff', model_diff(old_model, self._train_model, True), self.frame)

        # self._adjust_kl_scale(kl)
        self._eval_model = deepcopy(self._train_model).to(self.device_eval).eval()

    def _impala_step(self, batch, do_log=False):
        with torch.enable_grad():
            actor_params = AttrDict()
            if do_log:
                actor_params.logger = self.logger
                actor_params.cur_step = self.step

            actor_out = self._train_model(batch.states.reshape(-1, *batch.states.shape[2:]), **actor_params)

            batch.logits = actor_out.logits.reshape(*batch.states.shape[:2], *actor_out.logits.shape[1:])
            batch.state_values = actor_out.state_values.reshape(*batch.states.shape[:2])

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
        data.old_logp = pd.logp(data.actions, data.logits_old)
        data.logp = pd.logp(data.actions, data.logits)
        data.probs_ratio = (data.logp - data.old_logp).detach().mean(-1).exp()
        data.kl = kl = pd.kl(data.logits_old, data.logits)

        self._process_rewards(data)
        data.state_values = data.state_values[:-1]

        adv_u = data.advantages.unsqueeze(-1)
        entropy = pd.entropy(data.logits)
        loss_ent = self.entropy_loss_scale * -entropy
        loss_policy = -data.logp * adv_u
        loss_value = self.value_loss_scale * barron_loss(data.state_values, data.value_targets, *self.barron_alpha_c, reduce=False)

        # kl_targets = self.kl_target * adv_u.abs()
        # loss_kl = (kl - kl_targets).div(self.kl_target).pow(2).mul(0.001 * self.kl_scale * self.kl_target)
        loss_kl = (kl + (kl - 2 * self.kl_target).clamp(min=0).pow(2)).mul(0.01 * self.kl_scale)
        small_kl = kl.detach() < self.kl_target
        large_kl = kl.detach() > self.kl_target
        loss_kl[small_kl] = 0
        loss_ent[large_kl] = 0
        loss_policy[large_kl] = 0

        assert loss_ent.shape == loss_policy.shape, (loss_ent.shape, loss_policy.shape)
        assert loss_policy.shape == loss_kl.shape, (loss_policy.shape, loss_kl.shape)
        assert loss_policy.shape[:-1] == loss_value.shape, (loss_policy.shape, loss_value.shape)

        loss_ent = loss_ent.mean()
        loss_policy = loss_policy.mean()
        loss_kl = loss_kl.mean()

        # sum all losses
        total_loss = loss_policy + loss_value + loss_ent + loss_kl
        assert not np.isnan(total_loss.mean().item()) and not np.isinf(total_loss.mean().item()), \
            (loss_policy.mean().item(), loss_value.mean().item(), loss_ent.mean().item())

        if do_log and tag is not None:
            with torch.no_grad():
                self._log_training_data(data)
                self.logger.add_histogram('loss value hist' + tag, loss_value, self.frame)
                self.logger.add_histogram('loss ent hist' + tag, loss_ent, self.frame)
                self.logger.add_scalar('entropy' + tag, entropy.mean(), self.frame)
                self.logger.add_scalar('loss entropy' + tag, loss_ent.mean(), self.frame)
                self.logger.add_scalar('loss state value' + tag, loss_value.mean(), self.frame)
                ratio = (data.logp - data.old_logp).mean(-1)
                self.logger.add_histogram('ratio hist' + tag, ratio, self.frame)
                self.logger.add_scalar('ratio mean' + tag, ratio.mean(), self.frame)
                self.logger.add_scalar('ratio abs mean' + tag, ratio.abs().mean(), self.frame)
                self.logger.add_scalar('ratio abs max' + tag, ratio.abs().max(), self.frame)

        return total_loss, kl.mean()

    def _process_rewards(self, data, mean_norm=True):
        norm_rewards = self.reward_scale * data.rewards

        # calculate value targets and advantages
        value_targets, advantages, p = calc_vtrace(
            norm_rewards, data.state_values.detach(), data.dones, data.probs_ratio.detach(), data.kl.detach().mean(-1),
            self.reward_discount, self.vtrace_c_max, self.vtrace_p_max, self.vtrace_kl_limit)

        mean, square, iter = self._advantage_stats
        mean = self._advantage_momentum * mean + (1 - self._advantage_momentum) * advantages.mean().item()
        square = self._advantage_momentum * square + (1 - self._advantage_momentum) * advantages.pow(2).mean().item()
        iter += 1
        self._advantage_stats = (mean, square, iter)

        bias_corr = 1 - self._advantage_momentum ** iter
        mean = mean / bias_corr
        square = square / bias_corr

        if mean_norm:
            std = (square - mean ** 2) ** 0.5
            advantages = (advantages - mean) / max(std, 1e-3)
        else:
            rms = square ** 0.5
            advantages = advantages / max(rms, 1e-3)
        advantages = p * advantages

        data.value_targets, data.advantages, data.rewards = value_targets, advantages, norm_rewards

    def drop_collected_steps(self):
        self._prev_data = None