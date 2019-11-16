from functools import partial

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
from .utils import v_mpo_loss


class IMPALA(PPO):
    def __init__(self, *args,
                 replay_buf_size=128 * 1024,
                 replay_ratio=7,
                 min_replay_size=10000,
                 vtrace_c_max=1.0,
                 vtrace_p_max=1.0,
                 vtrace_kl_limit=0.3,
                 optimizer_factory=partial(RMSprop, lr=5e-4, eps=0.1),
                 grad_clip_norm=None,
                 eps_nu_alpha=(0.1, 0.005),
                 init_nu_alpha=(1.0, 5.0),
                 **kwargs):
        super().__init__(*args, optimizer_factory=optimizer_factory,
                         grad_clip_norm=grad_clip_norm, **kwargs)
        self.replay_buf_size = replay_buf_size
        self.replay_ratio = replay_ratio
        self.vtrace_c_max = vtrace_c_max
        self.vtrace_p_max = vtrace_p_max
        self.vtrace_kl_limit = vtrace_kl_limit
        self.min_replay_size = min_replay_size
        self.eps_nu_alpha = eps_nu_alpha
        self.nu = torch.scalar_tensor(init_nu_alpha[0], requires_grad=True)
        self.alpha = torch.scalar_tensor(init_nu_alpha[1], requires_grad=True)
        self._optimizer.add_param_group(dict(params=[self.nu, self.alpha]))

        # DataLoader limitation
        assert self.batch_size % self.horizon == 0, (self.batch_size, self.horizon)

        del self._steps_processor

        self._replay_buffer = ReplayBuffer(replay_buf_size)
        self._prev_data = None
        self._eval_steps = 0
        # self._advantage_stats = (0, 0, 0)
        # self._advantage_momentum = 0.99

        # self.eval_model_update_interval = 10
        # self._eval_no_copy_updates = 0
        # self.eps_nu = 0.1
        # self.eps_alpha = 0.005
        # self.nu = torch.scalar_tensor(1.0, device=self.device_train, requires_grad=True)
        # self.alpha = torch.scalar_tensor(5.0, device=self.device_train, requires_grad=True)
        # self._optimizer.add_param_group(dict(params=[self.nu, self.alpha]))

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
                    self._check_log()
                    self._train()
                    self._scheduler_step()

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
        if self.replay_ratio != 0 and len(self._replay_buffer) >= \
                max((self.horizon + 1) * self.num_actors * max(1, self.replay_ratio), self.min_replay_size):
            rand_samples = self._replay_buffer.sample(self.num_actors * self.replay_ratio, self.horizon + 1)
            return AttrDict({k: torch.cat([v1, v2], 1) for (k, v1), v2 in zip(rand_samples.items(), last_samples.values())})
        else:
            return AttrDict(last_samples)

    def _impala_update(self, data: AttrDict):
        if self.use_pop_art:
            self._train_model.heads.state_values.normalize(*self._pop_art.statistics)

        num_samples = (data.states.shape[0] - 1) * data.states.shape[1]
        num_rollouts = data.states.shape[1]

        data = AttrDict(states=data.states, logits_old=data.logits,
                        actions=data.actions, rewards=data.rewards, dones=data.dones)

        num_batches = num_samples // self.batch_size
        rand_idx = torch.randperm(num_rollouts, device=self.device_train).chunk(num_batches)
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
            self.logger.add_scalar('learning rate', self._learning_rate, self.frame)
            self.logger.add_scalar('clip mult', self._clip_mult, self.frame)
            self.logger.add_scalar('total loss', loss, self.frame)
            self.logger.add_scalar('kl', kl, self.frame)
            self.logger.add_scalar('kl scale', self.kl_scale, self.frame)
            self.logger.add_scalar('model abs diff', model_diff(old_model, self._train_model), self.frame)
            self.logger.add_scalar('model max diff', model_diff(old_model, self._train_model, True), self.frame)
            self.logger.add_scalar('nu', self.nu, self.frame)
            self.logger.add_scalar('alpha', self.alpha, self.frame)

        if self.use_pop_art:
            pa_mean, pa_std = self._pop_art.statistics
            value_targets = torch.cat(value_target_list, 0) * pa_std + pa_mean
            self._train_model.heads.state_values.unnormalize(pa_mean, pa_std)
            self._pop_art.update_statistics(value_targets)
            if self._do_log:
                self.logger.add_scalar('pop art mean', pa_mean, self.frame)
                self.logger.add_scalar('pop art std', pa_std, self.frame)

        # self._adjust_kl_scale(kl)
        NoisyLinear.randomize_network(self._train_model)

        self._copy_parameters(self._train_model, self._eval_model)
        # blend_models(self._train_model, self._eval_model, self.eval_model_blend)
        # self._eval_no_copy_updates += 1
        # if self._eval_no_copy_updates > self.eval_model_update_interval:
        #     self._eval_no_copy_updates = 0
        #     self._copy_parameters(self._train_model, self._eval_model)

    def _impala_step(self, batch, do_log):
        with torch.enable_grad():
            actor_params = AttrDict()
            if do_log:
                actor_params.logger = self.logger
                actor_params.cur_step = self.step

            actor_out = self._train_model(batch.states.reshape(-1, *batch.states.shape[2:]), **actor_params)
            with torch.no_grad():
                actor_out_policy = self._eval_model(batch.states.reshape(-1, *batch.states.shape[2:]), **actor_params)

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
        # for n, p in self._train_model.named_parameters():
        #     if p.grad is not None and (torch.isnan(p.grad.sum()) or torch.isinf(p.grad.sum())):
        #         print(n, p, p.grad)
        if self.grad_clip_norm is not None:
            clip_grad_norm_(self._train_model.parameters(), self.grad_clip_norm)
        self._optimizer.step()
        self._optimizer.zero_grad()

        self.nu.clamp_(min=1e-8)
        self.alpha.clamp_(min=1e-8)

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

        with torch.no_grad():
            self._process_rewards(data)
        data.state_values = data.state_values[:-1]

        data = AttrDict({k: v.flatten(end_dim=1) for k, v in data.items()})

        # print(f'kl {data.kl.shape}, logp {data.logp.shape}, old_logp {data.old_logp.shape}, '
        #       f'ratio {data.probs_ratio.shape}, adv {data.advantages.shape}, '
        #       f'values {data.state_values.shape}, targets {data.value_targets.shape}')

        kl_policy = pd.kl(data.logits_policy, data.logits)
        loss_policy, loss_nu, loss_alpha = v_mpo_loss(
            kl_policy, data.logp, data.advantages, self.nu, self.alpha, *self.eps_nu_alpha)

        # adv_clamp = data.advantages.clamp(-10, 10)
        # top_mask = adv_clamp >= adv_clamp.median()
        # top_advantages = adv_clamp[top_mask]
        # exp_top_advantages = top_advantages.div(self.nu).exp()
        # max_adv = adv_clamp.max()
        # softmax = adv_clamp.sub(max_adv).div(self.nu).exp().unsqueeze(-1) / \
        #           top_advantages.sub(max_adv).div(self.nu).exp().sum()
        # loss_policy = (softmax.detach() * -data.logp).mean(-1) * top_mask.float()
        # loss_nu = self.nu * self.eps_nu + self.nu * exp_top_advantages.mean().log()
        # loss_alpha = self.alpha * (self.eps_alpha - kl_policy.detach()) + self.alpha.detach() * kl_policy

        # for k, v in locals().items():
        #     if isinstance(v, torch.Tensor) and (torch.isnan(v.sum()) or torch.isinf(v.sum())):
        #         print(k, v)
        #         print('ac', adv_clamp, 'eta', exp_top_advantages, 'ta', top_advantages, 'tm', top_mask)


        # print(f'kl_policy {kl_policy.shape}, top_mask {top_mask.shape}, top_adv {top_advantages.shape}, '
        #       f'exp_top_adv {exp_top_advantages.shape}, '
        #       f'loss_policy {loss_policy.shape}, loss_nu {loss_nu.shape}, loss_alpha {loss_alpha.shape}')
        #
        # print(exp_top_advantages.div(exp_top_advantages.sum()).shape, data.logp[top_mask.unsqueeze(-1)].shape)

        # adv_u = data.advantages.unsqueeze(-1)
        entropy = pd.entropy(data.logits)
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
        # # loss_ent = loss_ent.mean()
        # loss_policy = loss_policy.sum()
        # loss_nu = loss_nu.mean()
        # loss_alpha = loss_alpha.mean()
        # loss_kl = loss_kl.mean()
        loss_value = loss_value.mean()

        assert loss_policy.shape == loss_value.shape == loss_nu.shape == loss_alpha.shape

        # sum all losses
        total_loss = loss_policy + loss_value + loss_nu + loss_alpha #+ loss_ent + loss_kl
        assert not np.isnan(total_loss.mean().item()) and not np.isinf(total_loss.mean().item()), \
            (loss_policy.mean().item(), loss_value.mean().item())

        if do_log and tag is not None:
            with torch.no_grad():
                self._log_training_data(data)
                self.logger.add_histogram('loss value hist' + tag, loss_value, self.frame)
                # self.logger.add_histogram('loss ent hist' + tag, loss_ent, self.frame)
                self.logger.add_scalar('entropy' + tag, entropy.mean(), self.frame)
                # self.logger.add_scalar('loss entropy' + tag, loss_ent.mean(), self.frame)
                self.logger.add_scalar('loss state value' + tag, loss_value.mean(), self.frame)
                self.logger.add_scalar('loss nu' + tag, loss_nu, self.frame)
                self.logger.add_scalar('loss alpha' + tag, loss_alpha, self.frame)
                ratio = (data.logp - data.old_logp).mean(-1)
                self.logger.add_histogram('ratio hist' + tag, ratio, self.frame)
                self.logger.add_scalar('ratio mean' + tag, ratio.mean(), self.frame)
                self.logger.add_scalar('ratio abs mean' + tag, ratio.abs().mean(), self.frame)
                self.logger.add_scalar('ratio abs max' + tag, ratio.abs().max(), self.frame)

        return total_loss, kl.mean()

    def _process_rewards(self, data, mean_norm=True):
        norm_rewards = self.reward_scale * data.rewards

        if self.use_pop_art:
            pa_mean, pa_std = self._pop_art.statistics

        state_values = data.state_values.detach() * pa_std + pa_mean if self.use_pop_art else data.state_values.detach()
        # calculate value targets and advantages
        value_targets, advantages = calc_vtrace(
            norm_rewards, state_values,
            data.dones, data.probs_ratio.detach(), data.kl.detach().sum(-1) / data.kl.shape[-1] ** 0.5,
            self.reward_discount, self.vtrace_c_max, self.vtrace_p_max, self.vtrace_kl_limit)

        if self.use_pop_art:
            value_targets = (value_targets - pa_mean) / pa_std

        # mean, square, iter = self._advantage_stats
        # mean = self._advantage_momentum * mean + (1 - self._advantage_momentum) * advantages.mean().item()
        # square = self._advantage_momentum * square + (1 - self._advantage_momentum) * advantages.pow(2).mean().item()
        # iter += 1
        # self._advantage_stats = (mean, square, iter)
        #
        # bias_corr = 1 - self._advantage_momentum ** iter
        # mean = mean / bias_corr
        # square = square / bias_corr
        #
        # # if mean_norm:
        # #     std = (square - mean ** 2) ** 0.5
        # #     advantages = (advantages - mean) / max(std, 1e-3)
        # # else:
        # rms = square ** 0.5
        # advantages = advantages / max(rms, 1e-3)
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        data.value_targets, data.advantages, data.rewards = value_targets, advantages, norm_rewards

    def drop_collected_steps(self):
        self._prev_data = None