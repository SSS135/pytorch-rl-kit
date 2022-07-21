import math
import pprint
from asyncio import Future
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from functools import partial
from typing import Optional, List, Callable

import numpy as np
import torch
import torch.autograd
import torch.optim as optim
from optfn.gradient_logger import log_gradients
from ..actors.fc_actors import FCFeatureExtractor
from ..common.squash import squash, unsquash
from ..algs.reward_weight_generator import RewardWeightGenerator
from rl_exp.noisy_linear import NoisyLinear
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

from .ppo import SchedulerManager, copy_state_dict, log_training_data
from .utils import impala_loss
from .running_norm import RunningNorm, GradRunningNorm
from .utils import v_mpo_loss
from .variable_replay_buffer import VariableReplayBuffer
from .variable_step_collector import VariableStepCollector
from ..actors.actors import Actor
from ..actors.fc_actors import create_ppo_fc_actor
from ..actors.utils import model_diff
from ..algs.utils import lerp_module_
from ..common.activation_norm import activation_norm_loss
from ..common.attr_dict import AttrDict
from ..common.data_loader import DataLoader
from ..common.gae import calc_vtrace, calc_upgo, calc_retrace
from ..common.pop_art import PopArt
from ..common.rl_base import RLBase, RLStepData
from itertools import chain


class IMPALA(RLBase):
    def __init__(self, observation_space, action_space,

                 use_pop_art=False,
                 reward_discount=0.99,
                 advantage_discount=1.0,
                 train_interval_frames=64 * 8,
                 train_horizon=64,
                 batch_size=64,
                 model_factory: Callable=create_ppo_fc_actor,
                 optimizer_factory=partial(optim.Adam, lr=3e-4),
                 value_loss_scale=1.0,
                 pg_loss_scale=1.0,
                 entropy_loss_scale=0.01,
                 cuda_eval=False,
                 cuda_train=False,
                 reward_scale=1.0,
                 lr_scheduler_factory=None,
                 clip_decay_factory=None,
                 entropy_decay_factory=None,

                 replay_buf_size=256 * 1024,
                 replay_ratio=7,
                 min_replay_size=10000,
                 upgo_scale=0.0,
                 grad_clip_norm=None,
                 kl_pull=0.5,
                 kl_limit=0.3,
                 replay_end_sampling_factor=0.1,
                 eval_model_blend=1.0,
                 memory_burn_in_steps=16,
                 activation_norm_scale=0.0,
                 reward_reweight_interval=40,
                 num_rewards=1,
                 squash_values=False,

                 ppo_iters=1,
                 ppo_policy_clip=None,
                 ppo_value_clip=None,

                 **kwargs):
        super().__init__(observation_space, action_space, **kwargs)
        self._init_args = locals()
        self.reward_discount = reward_discount
        self.advantage_discount = advantage_discount
        self.entropy_loss_scale = entropy_loss_scale
        self.train_interval_frames = train_interval_frames
        self.batch_size = batch_size
        self.device_eval = torch.device('cuda' if cuda_eval else 'cpu')
        self.device_train = torch.device('cuda' if cuda_train else 'cpu')
        self.grad_clip_norm = grad_clip_norm
        self.value_loss_scale = value_loss_scale
        self.pg_loss_scale = pg_loss_scale
        self.model_factory = model_factory
        self.optimizer_factory = optimizer_factory
        self.reward_scale = reward_scale
        self.use_pop_art = use_pop_art
        self.lr_scheduler_factory = lr_scheduler_factory
        self.replay_buf_size = replay_buf_size
        self.replay_ratio = replay_ratio
        self.upgo_scale = upgo_scale
        self.min_replay_size = min_replay_size
        self.replay_end_sampling_factor = replay_end_sampling_factor
        self.kl_pull = kl_pull
        self.kl_limit = kl_limit
        self.train_horizon = train_horizon
        self.eval_model_blend = eval_model_blend
        self.memory_burn_in_steps = memory_burn_in_steps
        self.activation_norm_scale = activation_norm_scale
        self.reward_reweight_interval = reward_reweight_interval
        self.num_rewards = num_rewards
        self.squash_values = squash_values
        self.ppo_iters = ppo_iters
        self.ppo_policy_clip = ppo_policy_clip
        self.ppo_value_clip = ppo_value_clip

        self._reward_weight_gen = RewardWeightGenerator(self.num_rewards)
        goal_size = self._reward_weight_gen.num_weights

        self._train_model: Actor = model_factory(observation_space, action_space, goal_size=goal_size)
        self._eval_model: Actor = model_factory(observation_space, action_space, goal_size=goal_size)
        self._target_model: Actor = model_factory(observation_space, action_space, goal_size=goal_size)
        if self.model_init_path is not None:
            self._train_model.load_state_dict(torch.load(self.model_init_path), True)
            print(f'loaded model {self.model_init_path}')
        copy_state_dict(self._train_model, self._eval_model)
        copy_state_dict(self._train_model, self._target_model)
        self._train_model = self._train_model.train().to(self.device_train, non_blocking=True)
        self._eval_model = self._eval_model.train().to(self.device_eval, non_blocking=True)
        self._target_model = self._target_model.train().to(self.device_train, non_blocking=True)

        if not self._train_model.is_recurrent:
            self.memory_burn_in_steps = memory_burn_in_steps = 0

        self._optimizer = optimizer_factory(self._train_model.parameters())
        self._scheduler = SchedulerManager(self._optimizer, lr_scheduler_factory, clip_decay_factory, entropy_decay_factory)
        self._pop_art = PopArt()

        assert not self._train_model.is_recurrent or self.memory_burn_in_steps < self.train_horizon
        # DataLoader limitation
        assert self.batch_size % self.train_horizon == 0, (self.batch_size, self.train_horizon)

        self._replay_buffer = VariableReplayBuffer(replay_buf_size, self.train_horizon, self.memory_burn_in_steps, self.replay_end_sampling_factor)
        self._step_collector = VariableStepCollector(
            self._eval_model, self._replay_buffer, self.device_eval, self._reward_weight_gen,
            self.reward_reweight_interval, self.disable_training)
        self._eval_no_copy_updates = 0
        self._adv_norm = RunningNorm(momentum=0.99, mean_norm=True)
        self._train_future: Optional[Future] = None
        self._data_future: Optional[Future] = None
        self._executor = ThreadPoolExecutor(max_workers=1, initializer=lambda: torch.set_num_threads(4))

    @property
    def _learning_rate(self):
        return self._optimizer.param_groups[0]['lr']

    @property
    def has_variable_actor_count_support(self):
        return True

    def _step(self, data: RLStepData) -> torch.Tensor:
        actions = self._step_collector.step(data)

        if not self.disable_training:
            if self._replay_buffer.avail_new_samples >= self.train_interval_frames:
                self._train()

        return self._eval_model.heads.logits.pd.postprocess_action(actions.cpu())

    def _train(self):
        self.frame_train = self.frame_eval

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

    def _create_data(self) -> AttrDict:
        roll_per_batch = self.batch_size // self.train_horizon

        def cat_replay(last, rand):
            # uniformly cat last and rand
            rand_roll = list(reversed(rand.unbind(dim=1)))
            last_roll = list(reversed(last.unbind(dim=1)))
            all = []
            while len(rand_roll) > 0 or len(last_roll) > 0:
                if len(last_roll) > 0:
                    all.append(last_roll.pop())
                for _ in range(max(1, self.replay_ratio)):
                    if len(rand_roll) > 0:
                        all.append(rand_roll.pop())
            return torch.stack(all, 1)

        enough_samples = self.replay_ratio != 0 and len(self._replay_buffer) >= \
                         max(2 * self.train_horizon * self.replay_ratio, self.min_replay_size)
        replay_ratio = self.replay_ratio if enough_samples else 0

        # (H, B, *)
        last_samples = self._replay_buffer.get_new_samples()
        n_new_rollouts = last_samples['dones'].shape[1]
        n_rand_rollouts = n_new_rollouts * replay_ratio
        n_rand_rollouts = math.ceil((n_rand_rollouts + n_new_rollouts) / roll_per_batch) * roll_per_batch - n_new_rollouts
        if n_rand_rollouts > 0:
            rand_samples = self._replay_buffer.sample(n_rand_rollouts)
            return AttrDict({k: cat_replay(last, rand)
                             for (k, rand), last in zip(rand_samples.items(), last_samples.values())})
        else:
            return AttrDict(last_samples)

    def _impala_update(self, data: AttrDict):
        self._apply_pop_art()

        num_samples = data.states.shape[0] * data.states.shape[1]
        num_rollouts = data.states.shape[1]
        assert num_samples > 0

        data = AttrDict(states=data.states, logits_replay=data.logits,
                        actions=data.actions, rewards=data.rewards, dones=data.dones,
                        **(dict(memory=data.memory) if self._train_model.is_recurrent else dict()))

        self._add_reward_weights(data)

        num_batches = max(1, num_samples // self.batch_size)
        rand_idx = torch.randperm(num_rollouts, device=self.device_train).chunk(num_batches)
        assert len(rand_idx) == num_batches, (len(rand_idx), num_batches, data.states.shape)

        old_model = {k: v.clone() for k, v in self._train_model.state_dict().items()}
        kls_target = []
        kls_replay = []
        value_target_list = []

        for ppo_iter in range(self.ppo_iters):
            with DataLoader(data, rand_idx, self.device_train, num_threads=2, dim=1) as data_loader:
                for batch_index in range(num_batches):
                    # prepare batch data
                    batch = AttrDict(data_loader.get_next_batch())
                    do_log = self._do_log and batch_index == num_batches - 1 and ppo_iter + 1 == self.ppo_iters
                    loss = self._impala_step(batch, do_log)
                    kls_target.append(batch.kl_target.mean().item())
                    kls_replay.append(batch.kl_replay.mean().item())
                    value_target_list.append(batch.state_values_td.detach())

        kl_target = np.mean(kls_target)
        kl_replay = np.mean(kls_replay)

        if self._do_log:
            self.logger.add_scalar('Optimizer/Learning Rate', self._learning_rate, self.frame_train)
            self.logger.add_scalar('Losses/Total Loss', loss, self.frame_train)
            self.logger.add_scalar('Stability/KL Blend', kl_target, self.frame_train)
            self.logger.add_scalar('Stability/KL Replay', kl_replay, self.frame_train)
            self.logger.add_scalar('Model Diff/Abs', model_diff(old_model, self._train_model), self.frame_train)
            self.logger.add_scalar('Model Diff/Max', model_diff(old_model, self._train_model, True), self.frame_train)

        self._unapply_pop_art(value_target_list)

        lerp_module_(self._target_model, self._train_model, self.eval_model_blend)

        copy_state_dict(self._train_model, self._eval_model)
        if self._eval_model.training:
            self._eval_model = self._eval_model.eval()
        NoisyLinear.randomize_network(self._eval_model)
        NoisyLinear.copy_noise(self._eval_model, self._train_model)
        NoisyLinear.copy_noise(self._eval_model, self._target_model)

    def _apply_pop_art(self):
        if self.use_pop_art:
            self._train_model.heads.state_values.normalize(*self._pop_art.statistics)

    def _unapply_pop_art(self, value_target_list: List[Tensor] = None):
        if self.use_pop_art:
            pa_mean, pa_std = self._pop_art.statistics
            self._train_model.heads.state_values.unnormalize(pa_mean, pa_std)
            if value_target_list is not None:
                for vtarg in value_target_list:
                    self._pop_art.update_statistics(vtarg * pa_std + pa_mean)
                if self._do_log:
                    self.logger.add_scalar('PopArt/Mean', pa_mean, self.frame_train)
                    self.logger.add_scalar('PopArt/Std', pa_std, self.frame_train)

    def _add_reward_weights(self, data: AttrDict):
        horizon, num_rollouts = data.rewards.shape[:2]
        data.reward_weights = self._reward_weight_gen.generate(num_rollouts)\
            .unsqueeze(0).repeat(horizon, 1, 1)
        data.rewards = (data.rewards * self._reward_weight_gen.get_true_weights(data.reward_weights)).sum(-1)
        assert data.rewards.shape == (horizon, num_rollouts)

    def _impala_step(self, batch, do_log):
        with torch.enable_grad():
            actor_params = AttrDict()
            if do_log:
                actor_params.logger = self.logger
                actor_params.cur_step = self.frame_train

            if self._train_model.is_recurrent:
                prev_rewards, prev_actions = batch.rewards[:-1], batch.actions[:-1]
                batch.update({k: v[1:] for k, v in batch.items()})
                actor_out = self._run_train_model_rnn(
                    batch.states, batch.memory, batch.dones, prev_rewards, prev_actions,
                    batch.reward_weights, actor_params, do_log)
            else:
                actor_out = self._run_train_model_feedforward(batch.states, batch.reward_weights, actor_params, do_log)
            batch.update(actor_out)

            for k, v in batch.items():
                batch[k] = v if k == 'states' or k.startswith('features') else v.cpu()

            # get loss
            loss = self._get_impala_loss(batch, do_log)
            loss = loss.mean()

        # optimize
        loss.backward()
        if self.grad_clip_norm is not None:
            clip_grad_norm_(self._train_model.parameters(), self.grad_clip_norm)
        self._optimizer.step()
        self._optimizer.zero_grad()

        return loss

    def _run_train_model_feedforward(self, states, reward_weights, actor_params, do_log):
        actor_out_train = self._train_model.run_fx(states, goal=reward_weights, **actor_params)
        with torch.no_grad():
            actor_out_target = self._target_model(states, goal=reward_weights)

        logits_features = self._train_model.head_features('logits', actor_out_train)
        value_features = self._train_model.head_features('state_values', actor_out_train)
        if do_log:
            value_features = log_gradients(value_features, self.logger, 'Values', self.frame_train)
            logits_features = log_gradients(logits_features, self.logger, 'Logits', self.frame_train)
        logits = self._train_model.heads.logits(logits_features)
        state_values = self._train_model.heads.state_values(value_features)

        return dict(
            logits=logits,
            state_values=state_values.squeeze(-1),
            logits_target=actor_out_target.logits,
            state_value_targets=actor_out_target.state_values.squeeze(-1),
            **(dict(features_raw=actor_out_train.features_raw_0) if hasattr(actor_out_train, 'features_raw_0') else {}),
        )

    def _run_train_model_rnn(self, states, memory, dones, prev_rewards, prev_actions, reward_weights, actor_params, do_log):
        with torch.no_grad():
            mem = memory[0]
            actor_out_train_burn_in = self._train_model.run_fx(
                states[:self.memory_burn_in_steps], memory=mem, dones=dones[:self.memory_burn_in_steps],
                prev_rewards=prev_rewards[:self.memory_burn_in_steps], prev_actions=prev_actions[:self.memory_burn_in_steps],
                goal=reward_weights[:self.memory_burn_in_steps])
            actor_out_target = self._target_model(
                states, memory=mem, dones=dones, prev_rewards=prev_rewards, prev_actions=prev_actions,
                goal=reward_weights)
        actor_out_train = self._train_model.run_fx(
            states[self.memory_burn_in_steps:], memory=actor_out_train_burn_in.memory.detach(), dones=dones[self.memory_burn_in_steps:],
            prev_rewards=prev_rewards[self.memory_burn_in_steps:], prev_actions=prev_actions[self.memory_burn_in_steps:],
            goal=reward_weights[self.memory_burn_in_steps:], **actor_params)

        logits_features = torch.cat([
            self._train_model.head_features('logits', actor_out_train_burn_in),
            self._train_model.head_features('logits', actor_out_train)], 0)
        value_features = torch.cat([
            self._train_model.head_features('state_values', actor_out_train_burn_in),
            self._train_model.head_features('state_values', actor_out_train)], 0)
        if do_log:
            value_features = log_gradients(value_features, self.logger, 'Values', self.frame_train)
            logits_features = log_gradients(logits_features, self.logger, 'Logits', self.frame_train)
        logits = self._train_model.heads.logits(logits_features)
        state_values = self._train_model.heads.state_values(value_features)

        return dict(
            logits=logits,
            state_values=state_values.squeeze(-1),
            logits_target=actor_out_target.logits,
            state_value_targets=actor_out_target.state_values.squeeze(-1),
            **(dict(features_raw=actor_out_train.features_raw_0) if hasattr(actor_out_train, 'features_raw_0') else {}),
        )

    def _get_impala_loss(self, data, do_log=False, tag=''):
        data.update({k: v[:-1] for k, v in data.items() if k != 'state_values'})

        pd = self._train_model.heads.logits.pd
        data.logp_replay = pd.logp(data.actions, data.logits_replay)
        data.logp_target = pd.logp(data.actions, data.logits_target)
        data.logp = pd.logp(data.actions, data.logits)
        data.logp_ratio_target = (data.logp - data.logp_target).sum(-1)
        data.prob_ratio_replay = (data.logp - data.logp_replay).sum(-1).exp()
        data.entropy = pd.entropy(data.logits)
        data.kl_replay = pd.kl(data.logits_replay, data.logits)
        data.kl_target = pd.kl(data.logits_target, data.logits)

        with torch.no_grad():
            self._process_rewards(data, do_log)
        data.state_values = data.state_values[:-1]

        for k, v in data.items():
            data[k] = v.flatten(end_dim=1)

        assert data.logp.shape[:-1] == data.prob_ratio_replay.shape
        assert data.kl_replay.shape[:-1] == data.prob_ratio_replay.shape
        assert data.advantages.shape == data.prob_ratio_replay.shape

        unclipped_policy_loss = data.logp_ratio_target * -data.advantages
        if self.ppo_policy_clip is not None:
            clipped_policy_loss = data.logp_ratio_target.clamp(-self.ppo_policy_clip, self.ppo_policy_clip) * -data.advantages
            loss_pg = torch.max(unclipped_policy_loss, clipped_policy_loss)
        else:
            loss_pg = unclipped_policy_loss
        assert loss_pg.shape == data.advantages.shape, (loss_pg.shape, data.advantages.shape)
        loss_pg = self.pg_loss_scale * loss_pg.mean()

        # loss_kl = self.kl_pull * data.kl_target.mean()
        loss_kl = self.kl_pull * 0.5 * (data.logits_target - data.logits).pow(2).mean()
        loss_ent = self.entropy_loss_scale * -data.entropy.mean()
        loss_ent = loss_ent.mean()

        # value loss
        values, values_blend, values_td = data.state_values, data.state_value_targets, data.state_values_td
        if self.ppo_value_clip is not None:
            v_pred_clipped = values_blend + (values - values_blend).clamp_(-self.ppo_value_clip, self.ppo_value_clip)
            vf_clip_loss = (v_pred_clipped - values_td).pow_(2)
            vf_nonclip_loss = (values - values_td).pow_(2)
            loss_state_value = self.value_loss_scale * 0.5 * torch.max(vf_nonclip_loss, vf_clip_loss)
            assert loss_state_value.shape == values.shape
        else:
            loss_state_value = self.value_loss_scale * (values - values_td).pow_(2).mul_(0.5)
        assert loss_state_value.shape == data.state_values.shape, (loss_state_value.shape, data.state_values.shape)
        loss_state_value = loss_state_value.mean()

        act_norm = activation_norm_loss(self._train_model).cpu().mean()
        if self.activation_norm_scale == 0:
            loss_act_norm = 0
        else:
            loss_act_norm = act_norm * max(self.activation_norm_scale, 1 - self.frame_train / 100_000)

        total_loss = loss_pg + loss_kl + loss_state_value + loss_ent + loss_act_norm
        assert not np.isnan(total_loss.mean().item()) and not np.isinf(total_loss.mean().item()), \
            ([x.mean().item() for x in (loss_pg, loss_kl, loss_state_value, loss_ent)])

        with torch.no_grad():
            if do_log:
                data = AttrDict(**data)
                if self.use_pop_art:
                    pa_mean, pa_std = self._pop_art.statistics
                    data.state_values = data.state_values * pa_std + pa_mean
                    data.state_values_td = data.state_values_td * pa_std + pa_mean
                if self.squash_values:
                    data.state_values = unsquash(data.state_values)
                    data.state_values_td = unsquash(data.state_values_td)
                log_training_data(self._do_log, self.logger, self.frame_train, self._train_model, data)

                ratio = data.logp - data.logp_target
                self.logger.add_scalar('Prob Ratio/Mean' + tag, ratio.mean(), self.frame_train)
                self.logger.add_scalar('Prob Ratio/Abs Mean' + tag, ratio.abs().mean(), self.frame_train)
                self.logger.add_scalar('Prob Ratio/Abs Max' + tag, ratio.abs().max(), self.frame_train)
                self.logger.add_scalar('VTrace P/Mean', data.vtrace_p.mean(), self.frame_train)
                self.logger.add_scalar('VTrace P/Above 0.75 Fraction', (data.vtrace_p > 0.75).float().mean(), self.frame_train)
                self.logger.add_scalar('Stability/Entropy' + tag, data.entropy.mean(), self.frame_train)
                self.logger.add_scalar('Losses/State Value' + tag, loss_state_value.mean(), self.frame_train)
                self.logger.add_histogram('Losses/Value Hist' + tag, loss_state_value, self.frame_train)
                self.logger.add_histogram('Losses/Ratio Hist' + tag, ratio, self.frame_train)
                # self.logger.add_scalar('Losses/Activation Norm', act_norm, self.frame_train)

        return total_loss

    def _process_rewards(self, data, do_log):
        if self.use_pop_art:
            pa_mean, pa_std = self._pop_art.statistics

        data.rewards = self.reward_scale * data.rewards
        state_values = data.state_values * pa_std + pa_mean if self.use_pop_art else data.state_values
        if self.squash_values:
            state_values = unsquash(state_values)
        probs_ratio = data.prob_ratio_replay
        kl_replay = data.kl_replay.mean(-1)

        # _, advantages, data.vtrace_p = calc_vtrace(
        #     data.rewards, state_values, data.dones, probs_ratio, kl_replay,
        #     self.reward_discount, self.kl_limit, lam=self.advantage_discount)

        state_values_td, advantages, data.vtrace_p = calc_vtrace(
            data.rewards, state_values, data.dones, probs_ratio, kl_replay,
            self.reward_discount, self.kl_limit, lam=1.0)

        advantages_upgo = calc_upgo(data.rewards, state_values, data.dones, self.reward_discount,
                                    lam=self.advantage_discount) - state_values[:-1]

        state_values_td = state_values_td[:-1]

        if do_log:
            self.logger.add_histogram('Advantages/Advantages', advantages, self.frame_train)
            self.logger.add_histogram('Advantages/UPGO', advantages_upgo, self.frame_train)
            self.logger.add_histogram('Values/Values', state_values, self.frame_train)
            self.logger.add_histogram('Values/Value Targets', state_values_td, self.frame_train)

        if self.squash_values:
            state_values_td = squash(state_values_td)
        if self.use_pop_art:
            state_values_td = (state_values_td - pa_mean) / pa_std
            if do_log:
                self.logger.add_scalar('Values/Values PopArt', data.state_values.mean(), self.frame_train)
                self.logger.add_scalar('Values/Value Targets PopArt', state_values_td.mean(), self.frame_train)

        assert data.vtrace_p.shape == advantages_upgo.shape
        advantages = self._adv_norm(advantages + self.upgo_scale * advantages_upgo)

        data.state_values_td = state_values_td
        data.advantages = advantages

    def drop_collected_steps(self):
        self._step_collector.drop_collected_steps()

    def _log_set(self):
        self.logger.add_text(self.__class__.__name__, pprint.pformat(self._init_args))
        self.logger.add_text('Model', str(self._train_model))

    def __getstate__(self):
        d = dict(self.__dict__)
        d['_logger'] = None
        return d

    def __setstate__(self, d):
        self.__dict__ = d