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
from ..common.gae import calc_vtrace
from ..models.utils import model_diff


class IMPALA(PPO):
    def __init__(self, *args, replay_buf_size=128 * 1024, off_policy_batches=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.replay_buf_size = replay_buf_size
        self.off_policy_batches = off_policy_batches

        assert self.batch_size % self.horizon == 0

        del self._steps_processor

        self._replay_buffer = ReplayBuffer(replay_buf_size)
        self._prev_data = None
        self._eval_steps = 0
        self._advantage_stats = (0, 1, 0)
        self._advantage_momentum = 0.99

    def _step(self, rewards, dones, states) -> torch.Tensor:
        with torch.no_grad():
            # run network
            ac_out = self._take_step(states.to(self.device_eval), dones)
            actions = self._eval_model.pd.sample(ac_out.logits).cpu()

            if not self.disable_training:
                if self._prev_data is not None and self._prev_data[0] is not None:
                    self._replay_buffer.push(**ac_out, states=states, actions=actions,
                                             rewards=self._prev_data[0], dones=self._prev_data[1])

                self._eval_steps += 1
                self._prev_data = (rewards, dones)

                if self._eval_steps >= self.horizon and len(self._replay_buffer) >= self.horizon * self.num_actors:
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
            self._check_save_model()

    def _create_data(self):
        # rand_actors = self.batch_size * self.off_policy_batches // self.horizon
        # rand_samples = self._replay_buffer.sample(rand_actors, self.horizon)
        # return AttrDict(rand_samples)

        last_samples = self._replay_buffer.get_last_samples(self.horizon)
        if self.off_policy_batches != 0:
            rand_actors = self.batch_size * self.off_policy_batches // self.horizon
            rand_samples = self._replay_buffer.sample(rand_actors, self.horizon)
            return AttrDict({k: torch.cat([v1, v2], 1) for (k, v1), v2 in zip(rand_samples.items(), last_samples.values())})
        else:
            return AttrDict(last_samples)

    def _impala_update(self, data: AttrDict):
        num_samples = data.states.shape[0] * data.states.shape[1]
        num_rollouts = data.states.shape[1]

        data = AttrDict(states=data.states, old_logits=data.logits, actions=data.actions,
                        old_state_values=data.state_values, rewards=data.rewards, dones=data.dones)

        num_batches = num_samples // self.batch_size
        rand_idx = torch.randperm(num_rollouts, device=self.device_train).chunk(num_batches)

        old_model = deepcopy(self._train_model)
        kl_list = []

        with DataLoader(data, rand_idx, self.device_train, 4, dim=1) as data_loader:
            for loader_iter in range(num_batches):
                # prepare batch data
                batch = AttrDict(data_loader.get_next_batch())
                if loader_iter == num_batches - 1:
                    self._train_model.set_log(self.logger, self._do_log, self.step)

                loss, kl = self._impala_step(batch)
                kl_list.append(kl)

                self._train_model.set_log(self.logger, False, self.step)

        kl = np.mean(kl_list)

        if self._do_log:
            self.logger.add_scalar('learning rate', self._learning_rate, self.frame)
            self.logger.add_scalar('clip mult', self._clip_mult, self.frame)
            self.logger.add_scalar('total loss', loss, self.frame)
            self.logger.add_scalar('kl', kl, self.frame)
            self.logger.add_scalar('kl scale', self.kl_scale, self.frame)
            self.logger.add_scalar('model abs diff', model_diff(old_model, self._train_model), self.frame)
            self.logger.add_scalar('model max diff', model_diff(old_model, self._train_model, True), self.frame)

        self._adjust_kl_scale(kl)
        self._eval_model = deepcopy(self._train_model).to(self.device_eval).eval()

    def _impala_step(self, data):
        with torch.enable_grad():
            actor_out = self._train_model(data.states.reshape(-1, *data.states.shape[2:]))
            data.logits, data.state_values = [x.view(*data.states.shape[:2], *x.shape[1:])
                                              for x in (actor_out.logits, actor_out.state_values)]
            # get loss
            loss, kl = self._get_impala_loss(data)
            loss = loss.mean()

        kl = kl.item()

        # optimize
        loss.backward()
        if self.grad_clip_norm is not None:
            clip_grad_norm_(self._train_model.parameters(), self.grad_clip_norm)
        self._optimizer.step()
        self._optimizer.zero_grad()

        return loss, kl

    def _get_impala_loss(self, data, pd=None, tag=''):
        """
        Single iteration of PPO algorithm.
        Returns: Total loss and KL divergence.
        """

        if pd is None:
            pd = self._train_model.pd

        # action probability ratio
        # log probabilities used for better numerical stability
        data.old_logp = pd.logp(data.actions, data.old_logits).mean(-1)
        data.logp = pd.logp(data.actions, data.logits).mean(-1)
        data.probs_ratio = torch.exp(data.logp - data.old_logp.detach())

        self._process_rewards(data)

        entropy = pd.entropy(data.logits)
        kl = pd.kl(data.old_logits, data.logits)
        loss_kl = self.kl_scale * (kl + (kl - 2 * self.kl_target).clamp(0, 1e6).pow(2)).mean()

        policy_loss = (-data.logp * data.advantages).mean()
        loss_ent = self.entropy_loss_scale * -entropy.mean()
        loss_value = self.value_loss_scale * barron_loss(data.state_values, data.returns, *self.barron_alpha_c)

        # sum all losses
        total_loss = policy_loss + loss_value + loss_ent + loss_kl
        assert not np.isnan(total_loss.mean().item()) and not np.isinf(total_loss.mean().item()), \
            (policy_loss.mean().item(), loss_value.mean().item(), loss_ent.mean().item())

        if self._train_model.do_log and tag is not None:
            with torch.no_grad():
                self._log_training_data(data)
                self.logger.add_histogram('loss value' + tag, loss_value, self.frame)
                self.logger.add_histogram('loss ent' + tag, loss_ent, self.frame)
                self.logger.add_scalar('entropy' + tag, entropy.mean(), self.frame)
                self.logger.add_scalar('loss entropy' + tag, loss_ent.mean(), self.frame)
                self.logger.add_scalar('loss value' + tag, loss_value.mean(), self.frame)
                ratio = data.logp - data.old_logp.detach()
                self.logger.add_histogram('ratio' + tag, ratio, self.frame)
                self.logger.add_scalar('ratio mean' + tag, ratio.mean(), self.frame)
                self.logger.add_scalar('ratio abs mean' + tag, ratio.abs().mean(), self.frame)
                self.logger.add_scalar('ratio abs max' + tag, ratio.abs().max(), self.frame)

        return total_loss, kl.mean()

    def _process_rewards(self, data, mean_norm=True):
        state_values = data.state_values
        data.update({k: v[:-1] for k, v in data.items()})

        norm_rewards = self.reward_scale * data.rewards

        # calculate returns and advantages
        returns, advantages = calc_vtrace(
            norm_rewards, state_values.detach(), data.dones, data.probs_ratio.detach(), self.reward_discount)

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
        advantages = barron_loss_derivative(advantages, *self.barron_alpha_c)

        data.returns, data.advantages, data.rewards = returns, advantages, norm_rewards

    def drop_collected_steps(self):
        self._prev_data = None