import pprint
from asyncio import Future
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from functools import partial
from typing import Optional

import gym
import numpy as np
import torch
import torch.autograd
import torch.optim as optim
from optfn.gradient_logger import log_gradients
from ppo_pytorch.algs.ppo import SchedulerManager, copy_state_dict, log_training_data
from torch.nn.utils import clip_grad_norm_

from .replay_buffer import ReplayBuffer
from .utils import RunningNorm, scaled_impala_loss
from .utils import v_mpo_loss
from ..actors import create_ppo_fc_actor, Actor
from ppo_pytorch.common.activation_norm import activation_norm_loss
from ..actors.utils import model_diff
from ..algs.utils import lerp_module_
from ..common.attr_dict import AttrDict
from ..common.barron_loss import barron_loss
from ..common.data_loader import DataLoader
from ..common.gae import calc_vtrace
from ..common.pop_art import PopArt
from ..common.rl_base import RLBase


class LossType(Enum):
    v_mpo = 'v_mpo'
    impala = 'impala'


class IMPALA(RLBase):
    def __init__(self, observation_space, action_space,

                 use_pop_art=False,
                 reward_discount=0.99,
                 horizon=64,
                 batch_size=64,
                 model_factory=create_ppo_fc_actor,
                 optimizer_factory=partial(optim.Adam, lr=3e-4),
                 value_loss_scale=0.5,
                 entropy_loss_scale=0.01,
                 cuda_eval=False,
                 cuda_train=False,
                 reward_scale=1.0,
                 barron_alpha_c=(1.5, 1),
                 lr_scheduler_factory=None,
                 clip_decay_factory=None,
                 entropy_decay_factory=None,

                 replay_buf_size=256 * 1024,
                 replay_ratio=7,
                 min_replay_size=10000,
                 vtrace_max_ratio=1.0,
                 vtrace_kl_limit=0.2,
                 upgo_scale=0.0,
                 grad_clip_norm=None,
                 kl_pull=0.1,
                 kl_limit=0.2,
                 replay_end_sampling_factor=0.1,
                 train_horizon=None,
                 loss_type='impala',
                 eval_model_blend=0.1,
                 memory_burn_in_steps=16,
                 activation_norm_scale=0.01,

                 **kwargs):
        super().__init__(observation_space, action_space, **kwargs)
        self._init_args = locals()
        self.reward_discount = reward_discount
        self.entropy_loss_scale = entropy_loss_scale
        self.horizon = horizon
        self.batch_size = batch_size
        self.device_eval = torch.device('cuda' if cuda_eval else 'cpu')
        self.device_train = torch.device('cuda' if cuda_train else 'cpu')
        self.grad_clip_norm = grad_clip_norm
        self.value_loss_scale = value_loss_scale
        self.model_factory = model_factory
        self.optimizer_factory = optimizer_factory
        self.reward_scale = reward_scale
        self.barron_alpha_c = barron_alpha_c
        self.use_pop_art = use_pop_art
        self.lr_scheduler_factory = lr_scheduler_factory
        self.replay_buf_size = replay_buf_size
        self.replay_ratio = replay_ratio
        self.vtrace_max_ratio = vtrace_max_ratio
        self.vtrace_kl_limit = vtrace_kl_limit
        self.upgo_scale = upgo_scale
        self.min_replay_size = min_replay_size
        self.replay_end_sampling_factor = replay_end_sampling_factor
        self.kl_pull = kl_pull
        self.kl_limit = kl_limit
        self.train_horizon = self.horizon if train_horizon is None else train_horizon
        self.eval_model_blend = eval_model_blend
        self.memory_burn_in_steps = memory_burn_in_steps
        self.activation_norm_scale = activation_norm_scale

        self._train_model: Actor = model_factory(observation_space, action_space)
        self._eval_model: Actor = model_factory(observation_space, action_space)
        if self.model_init_path is not None:
            self._train_model.load_state_dict(torch.load(self.model_init_path), True)
            print(f'loaded model {self.model_init_path}')
        copy_state_dict(self._train_model, self._eval_model)
        self._train_model = self._train_model.train().to(self.device_train, non_blocking=True)
        self._eval_model = self._eval_model.eval().to(self.device_eval, non_blocking=True)

        self._optimizer = optimizer_factory(self._train_model.parameters())
        self._scheduler = SchedulerManager(self._optimizer, lr_scheduler_factory, clip_decay_factory, entropy_decay_factory)
        self._last_model_save_frame = 0
        self._pop_art = PopArt()
        self._first_pop_art_update = True
        self._target_step = 0

        assert isinstance(loss_type, str) or isinstance(loss_type, list) or isinstance(loss_type, tuple)
        self.loss_type = (loss_type,) if isinstance(loss_type, str) else loss_type
        self.loss_type = [LossType[c] for c in self.loss_type]
        assert len(set(self.loss_type) - set(c for c in LossType)) == 0

        assert self.memory_burn_in_steps < self.train_horizon
        # DataLoader limitation
        assert self.batch_size % self.train_horizon == 0 and self.horizon % self.train_horizon == 0, (self.batch_size, self.horizon)

        self._replay_buffer = ReplayBuffer(replay_buf_size)
        self._prev_data = None
        self._eval_steps = 0
        self._eval_no_copy_updates = 0
        self._adv_norm = RunningNorm(momentum=0.95, mean_norm=False)
        self._train_future: Optional[Future] = None
        self._data_future: Optional[Future] = None
        self._executor = ThreadPoolExecutor(max_workers=1, initializer=lambda: torch.set_num_threads(1))

        self._target_model = self.model_factory(self.observation_space, self.action_space).to(self.device_train).train()
        self._target_model.load_state_dict(self._train_model.state_dict())
        self._eval_model = self._eval_model.eval()

    @property
    def _learning_rate(self):
        return self._optimizer.param_groups[0]['lr']

    def _step(self, rewards, dones, states) -> torch.Tensor:
        with torch.no_grad():
            if self._eval_model.is_recurrent:
                input_memory = self._prev_data.memory if self._prev_data is not None else None
                dones_t = dones.unsqueeze(0).to(self.device_eval) if dones is not None else \
                    torch.zeros((1, self.num_actors), device=self.device_eval)
                ac_out = self._eval_model(states.unsqueeze(0).to(self.device_eval), memory=input_memory, dones=dones_t)
                ac_out = AttrDict({k: v.squeeze(0) for k, v in ac_out.items()})
                if input_memory is None:
                    input_memory = ac_out.memory
            else:
                ac_out = self._eval_model(states.to(self.device_eval))

            ac_out.state_values = ac_out.state_values.squeeze(-1)
            actions = self._eval_model.heads.logits.pd.sample(ac_out.logits).cpu()
            assert not torch.isnan(actions.sum())

            self._eval_steps += 1

            if not self.disable_training:
                if self._prev_data is not None and rewards is not None:
                    if self._eval_model.is_recurrent:
                        self._prev_data.memory = self._prev_data.input_memory
                        del self._prev_data['input_memory']
                    self._replay_buffer.push(rewards=rewards, dones=dones, **self._prev_data)

                self._prev_data = AttrDict(**ac_out, states=states, actions=actions)
                if self._eval_model.is_recurrent:
                    self._prev_data.input_memory = input_memory

                if self._eval_steps >= self.horizon + 2:
                    self._eval_steps = 0
                    self._train()

            return self.limit_actions(actions)

    def limit_actions(self, actions):
        if isinstance(self.action_space, gym.spaces.Box):
            return actions.clamp(-3, 3) / 3
        else:
            assert isinstance(self.action_space, gym.spaces.Discrete) or \
                   isinstance(self.action_space, gym.spaces.MultiDiscrete)
            return actions

    def _train(self):
        self.step_train = self.step_eval

        # data = self._create_data()
        # self._train_async(data)

        # self._train_async(data)
        # if self._data_future is not None:
        #     data = self._data_future.result()
        # else:
        #     data = self._create_data()
        # self._data_future = self._executor.submit(self._create_data)

        data = self._create_data()
        if self._train_future is not None:
            self._train_future.result()
        self._train_future = self._executor.submit(self._train_async, data)

    def _train_async(self, data):
        with torch.no_grad():
            self._check_log()
            self._impala_update(data)
            self._model_saver.check_save_model(self._train_model, self.frame_train)
            self._scheduler.step(self.frame_train)

    def _create_data(self):
        def cat_replay(last, rand):
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
        eval_model = self._eval_model

        if self.use_pop_art:
            self._train_model.heads.state_values.normalize(*self._pop_art.statistics)

        num_samples = data.states.shape[0] * data.states.shape[1]
        num_rollouts = data.states.shape[1]

        data = AttrDict(states=data.states, logits_old=data.logits,
                        actions=data.actions, rewards=data.rewards, dones=data.dones,
                        **(dict(memory=data.memory) if self._train_model.is_recurrent else dict()))

        num_batches = max(1, num_samples // self.batch_size)
        rand_idx = torch.arange(num_rollouts, device=self.device_train).chunk(num_batches)
        assert len(rand_idx) == num_batches

        old_model = {k: v.clone() for k, v in self._train_model.state_dict().items()}
        kls_policy = []
        kls_replay = []
        value_target_list = []

        with DataLoader(data, rand_idx, self.device_train, num_threads=2, dim=1) as data_loader:
            for batch_index in range(num_batches):
                # prepare batch data
                batch = AttrDict(data_loader.get_next_batch())
                loss = self._impala_step(batch, self._do_log and batch_index == num_batches - 1)
                kls_policy.append(batch.kl_policy.mean().item())
                kls_replay.append(batch.kl_replay.mean().item())
                value_target_list.append(batch.value_targets.detach())

        kl_policy = np.mean(kls_policy)
        kl_replay = np.mean(kls_replay)

        if self._do_log:
            self.logger.add_scalar('Optimizer/Learning Rate', self._learning_rate, self.frame_train)
            if loss is not None:
                self.logger.add_scalar('Losses/Total Loss', loss, self.frame_train)
            self.logger.add_scalar('Stability/KL Blend', kl_policy, self.frame_train)
            self.logger.add_scalar('Stability/KL Replay', kl_replay, self.frame_train)
            self.logger.add_scalar('Model Diff/Abs', model_diff(old_model, self._train_model), self.frame_train)
            self.logger.add_scalar('Model Diff/Max', model_diff(old_model, self._train_model, True), self.frame_train)

        if self.use_pop_art:
            pa_mean, pa_std = self._pop_art.statistics
            if len(value_target_list) != 0:
                value_targets = torch.cat(value_target_list, 0) * pa_std + pa_mean
            self._train_model.heads.state_values.unnormalize(pa_mean, pa_std)
            if len(value_target_list) != 0:
                self._pop_art.update_statistics(value_targets)
            if self._do_log:
                self.logger.add_scalar('PopArt/Mean', pa_mean, self.frame_train)
                self.logger.add_scalar('PopArt/Std', pa_std, self.frame_train)

        copy_state_dict(self._train_model, eval_model)
        lerp_module_(self._target_model, self._train_model, self.eval_model_blend)

    def _impala_step(self, batch, do_log):
        with torch.enable_grad():
            actor_params = AttrDict()
            if do_log:
                actor_params.logger = self.logger
                actor_params.cur_step = self.frame_train

            if self._train_model.is_recurrent:
                actor_out = self._run_train_model_rnn(batch.states, batch.memory, batch.dones, actor_params)
            else:
                actor_out = self._run_train_model_fc(batch.states, actor_params)
            batch.update(actor_out)

            for k, v in list(batch.items()):
                batch[k] = v if k == 'states' else v.cpu()

            if do_log:
                batch.state_values = log_gradients(batch.state_values, self.logger, 'Values', self.frame_train)
                batch.logits = log_gradients(batch.logits, self.logger, 'Logits', self.frame_train)

            # get loss
            loss = self._get_impala_loss(batch, do_log)
            if loss is None:
                return None
            act_norm_loss = activation_norm_loss(self._train_model).cpu()
            loss = loss.mean() + self.activation_norm_scale * act_norm_loss

        if do_log:
            self.logger.add_scalar('Losses/Activation Norm', act_norm_loss, self.frame_train)

        # optimize
        loss.backward()
        if self.grad_clip_norm is not None:
            clip_grad_norm_(self._train_model.parameters(), self.grad_clip_norm)
        self._optimizer.step()
        self._optimizer.zero_grad()

        return loss

    def _run_train_model_fc(self, states, actor_params):
        input_states = states.reshape(-1, *states.shape[2:])
        actor_out = self._train_model(input_states, **actor_params)
        with torch.no_grad():
            actor_out_policy = self._target_model(input_states)

        logits = actor_out.logits\
            .reshape(*states.shape[:2], *actor_out.logits.shape[1:])
        logits_policy = actor_out_policy.logits\
            .reshape(*states.shape[:2], *actor_out.logits.shape[1:])
        state_values = actor_out.state_values\
            .reshape(*states.shape[:2])

        return dict(logits=logits, logits_policy=logits_policy, state_values=state_values)

    def _run_train_model_rnn(self, states, memory, dones, actor_params):
        with torch.no_grad():
            actor_out_burn_in = self._train_model(states[:self.memory_burn_in_steps], memory=memory[0], dones=dones[:self.memory_burn_in_steps])
            actor_out_policy = self._target_model(states, memory=memory[0], dones=dones)
        actor_out = self._train_model(states[self.memory_burn_in_steps:], memory=actor_out_burn_in.memory, dones=dones[self.memory_burn_in_steps:], **actor_params)

        return dict(
            logits=torch.cat([actor_out_burn_in.logits, actor_out.logits], 0),
            state_values=torch.cat([actor_out_burn_in.state_values, actor_out.state_values], 0).squeeze(-1),
            logits_policy=actor_out_policy.logits)

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
        data.logp_old = pd.logp(data.actions, data.logits_old)
        data.logp_policy = pd.logp(data.actions, data.logits_policy)
        data.logp = pd.logp(data.actions, data.logits)
        data.probs_ratio = (data.logp.detach() - data.logp_old).exp()
        data.kl_replay = pd.kl(data.logits, data.logits_old)

        with torch.no_grad():
            self._process_rewards(data, do_log)
        data.state_values = data.state_values[:-1]

        for k, v in data.items():
            data[k] = v.flatten(end_dim=1)

        data.kl_policy = kl_policy = pd.kl(data.logits, data.logits_policy)
        if LossType.v_mpo in self.loss_type:
            losses = v_mpo_loss(
                kl_policy, data.logp, data.advantages, data.advantages_upgo, data.vtrace_p, self.kl_pull)
            if losses is None:
                return None
            loss_policy, loss_kl = losses
            loss_policy = loss_policy + loss_kl
        elif LossType.impala in self.loss_type:
            losses = scaled_impala_loss(
                kl_policy, data.logp, data.advantages, data.advantages_upgo,
                data.vtrace_p, self.kl_pull, self.kl_limit)
            if losses is None:
                return None
            loss_policy, loss_kl = losses
            loss_policy = loss_policy + loss_kl

            if do_log:
                self.logger.add_scalar('Losses/Alpha' + tag, loss_kl, self.frame_train)

        entropy = pd.entropy(data.logits)
        loss_ent = self.entropy_loss_scale * -entropy.mean()
        loss_value = self.value_loss_scale * barron_loss(data.state_values, data.value_targets, *self.barron_alpha_c, reduce=False)

        assert loss_value.shape == data.state_values.shape, (loss_value.shape, data.state_values.shape)

        loss_ent = loss_ent.mean()
        loss_value = loss_value.mean()

        # sum all losses
        total_loss = loss_policy + loss_value + loss_ent
        assert not np.isnan(total_loss.mean().item()) and not np.isinf(total_loss.mean().item()), \
            (loss_policy.mean().item(), loss_value.mean().item())

        with torch.no_grad():
            if do_log:
                if self.use_pop_art:
                    pa_mean, pa_std = self._pop_art.statistics
                    data = AttrDict(**data)
                    data.state_values = data.state_values * pa_std + pa_mean
                    data.value_targets = data.value_targets * pa_std + pa_mean
                log_training_data(self._do_log, self.logger, self.frame_train, self._train_model, data)
                ratio = (data.logp - data.logp_policy).exp() - 1
                self.logger.add_scalar('Prob Ratio/Mean' + tag, ratio.mean(), self.frame_train)
                self.logger.add_scalar('Prob Ratio/Abs Mean' + tag, ratio.abs().mean(), self.frame_train)
                self.logger.add_scalar('Prob Ratio/Abs Max' + tag, ratio.abs().max(), self.frame_train)
                self.logger.add_scalar('VTrace P/Mean', data.vtrace_p.mean(), self.frame_train)
                self.logger.add_scalar('VTrace P/Above 0.25 Fraction', (data.vtrace_p > 0.25).float().mean(), self.frame_train)
                self.logger.add_scalar('Stability/Entropy' + tag, entropy.mean(), self.frame_train)
                self.logger.add_scalar('Losses/State Value' + tag, loss_value.mean(), self.frame_train)
                if LossType.v_mpo is self.loss_type:
                    self.logger.add_scalar('Losses/Alpha' + tag, loss_kl, self.frame_train)
                self.logger.add_histogram('Losses/Value Hist' + tag, loss_value, self.frame_train)
                self.logger.add_histogram('Losses/Ratio Hist' + tag, ratio, self.frame_train)

        return total_loss

    def _process_rewards(self, data, do_log, mean_norm=True):
        norm_rewards = self.reward_scale * data.rewards

        if self.use_pop_art:
            pa_mean, pa_std = self._pop_art.statistics

        state_values = data.state_values.detach() * pa_std + pa_mean if self.use_pop_art else data.state_values.detach()
        # calculate value targets and advantages
        value_targets, advantages, advantages_upgo, p = calc_vtrace(
            norm_rewards, state_values,
            data.dones, data.probs_ratio.detach().mean(-1), data.kl_replay.detach().mean(-1),
            self.reward_discount, self.vtrace_max_ratio, self.vtrace_kl_limit)

        if do_log:
            self.logger.add_scalar('Advantages/Mean', advantages.mean(), self.frame_train)
            self.logger.add_scalar('Advantages/RMS', advantages.pow(2).mean().sqrt(), self.frame_train)
            self.logger.add_scalar('Advantages/Std', advantages.std(), self.frame_train)

        advantages_upgo *= self.upgo_scale
        if self.use_pop_art:
            value_targets = (value_targets - pa_mean) / pa_std
            if LossType.impala is self.loss_type:
                advantages /= pa_std
                advantages_upgo /= pa_std

        # if LossType.impala is self.loss_type:
        advantages = self._adv_norm(advantages)
        advantages_upgo = self._adv_norm(advantages_upgo, update_stats=False)

        data.vtrace_p, data.advantages_upgo = p, advantages_upgo
        data.value_targets, data.advantages, data.rewards = value_targets, advantages, norm_rewards

    def drop_collected_steps(self):
        self._prev_data = None

    def _log_set(self):
        self.logger.add_text(self.__class__.__name__, pprint.pformat(self._init_args))
        self.logger.add_text('Model', str(self._train_model))

    def __getstate__(self):
        d = dict(self.__dict__)
        d['_logger'] = None
        return d

    def __setstate__(self, d):
        self.__dict__ = d