import pprint
from asyncio import Future
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from functools import partial
from typing import Optional, List, Callable
import math

import gym
import numpy as np
import torch
import torch.autograd
import torch.optim as optim
from optfn.gradient_logger import log_gradients
from ppo_pytorch.actors.fc_actors import FCFeatureExtractor
from ppo_pytorch.algs.reward_weight_generator import RewardWeightGenerator

from .ppo import SchedulerManager, copy_state_dict, log_training_data
from torch.nn.utils import clip_grad_norm_

from .variable_step_collector import VariableStepCollector
from .variable_replay_buffer import VariableReplayBuffer
from .utils import RunningNorm, scaled_impala_loss
from .utils import v_mpo_loss
from ..actors.fc_actors import create_ppo_fc_actor
from ..actors.actors import Actor
from ..common.activation_norm import activation_norm_loss
from ..actors.utils import model_diff
from ..algs.utils import lerp_module_
from ..common.attr_dict import AttrDict
from ..common.barron_loss import barron_loss
from ..common.data_loader import DataLoader
from ..common.gae import calc_vtrace, calc_value_targets, calc_upgo
from ..common.pop_art import PopArt
from ..common.rl_base import RLBase, RLStepData
from torch import Tensor
import kornia


class LossType(Enum):
    v_mpo = 'v_mpo'
    impala = 'impala'


class IMPALA(RLBase):
    def __init__(self, observation_space, action_space,

                 use_pop_art=False,
                 reward_discount=0.99,
                 train_interval_frames=64 * 8,
                 train_horizon=64,
                 batch_size=64,
                 model_factory: Callable=create_ppo_fc_actor,
                 optimizer_factory=partial(optim.Adam, lr=3e-4),
                 value_loss_scale=0.5,
                 q_loss_scale=1.0,
                 dpg_loss_scale=0.0,
                 pg_loss_scale=1.0,
                 entropy_loss_scale=0.01,
                 cuda_eval=False,
                 cuda_train=False,
                 reward_scale=1.0,
                 barron_alpha_c=(1.5, 1),
                 lr_scheduler_factory=None,
                 clip_decay_factory=None,
                 entropy_decay_factory=None,
                 random_crop_obs=False,

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
                 loss_type='impala',
                 eval_model_blend=0.1,
                 memory_burn_in_steps=16,
                 activation_norm_scale=0.003,
                 reward_reweight_interval=40,
                 num_rewards=1,

                 **kwargs):
        super().__init__(observation_space, action_space, **kwargs)
        self._init_args = locals()
        self.reward_discount = reward_discount
        self.entropy_loss_scale = entropy_loss_scale
        self.train_interval_frames = train_interval_frames
        self.batch_size = batch_size
        self.device_eval = torch.device('cuda' if cuda_eval else 'cpu')
        self.device_train = torch.device('cuda' if cuda_train else 'cpu')
        self.grad_clip_norm = grad_clip_norm
        self.value_loss_scale = value_loss_scale
        self.q_loss_scale = q_loss_scale
        self.dpg_loss_scale = dpg_loss_scale
        self.pg_loss_scale = pg_loss_scale
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
        self.train_horizon = train_horizon
        self.eval_model_blend = eval_model_blend
        self.memory_burn_in_steps = memory_burn_in_steps
        self.activation_norm_scale = activation_norm_scale
        self.reward_reweight_interval = reward_reweight_interval
        self.num_rewards = num_rewards
        self.random_crop_obs = random_crop_obs

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
        self._eval_model = self._eval_model.eval().to(self.device_eval, non_blocking=True)
        self._target_model = self._target_model.train().to(self.device_train, non_blocking=True)

        self._optimizer = optimizer_factory(self._train_model.parameters())
        self._scheduler = SchedulerManager(self._optimizer, lr_scheduler_factory, clip_decay_factory, entropy_decay_factory)
        self._last_model_save_frame = 0
        self._pop_art = PopArt()
        self._first_pop_art_update = True
        self._target_step = 0
        self._no_blend_batches = 0

        assert isinstance(loss_type, str) or isinstance(loss_type, list) or isinstance(loss_type, tuple)
        self.loss_type = (loss_type,) if isinstance(loss_type, str) else loss_type
        self.loss_type = [LossType[c] for c in self.loss_type]
        assert len(set(self.loss_type) - set(c for c in LossType)) == 0

        assert not self._train_model.is_recurrent or self.memory_burn_in_steps < self.train_horizon
        # DataLoader limitation
        assert self.batch_size % self.train_horizon == 0, (self.batch_size, self.train_horizon)

        self._replay_buffer = VariableReplayBuffer(replay_buf_size, self.train_horizon, self.replay_end_sampling_factor)
        self._step_collector = VariableStepCollector(
            self._eval_model, self._replay_buffer, self.device_eval, self._reward_weight_gen,
            self.reward_reweight_interval, self.disable_training)
        self._eval_no_copy_updates = 0
        self._adv_norm = RunningNorm(momentum=0.99, mean_norm=False)
        self._train_future: Optional[Future] = None
        self._data_future: Optional[Future] = None
        self._executor = ThreadPoolExecutor(max_workers=1, initializer=lambda: torch.set_num_threads(4))
        self._crop = None

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

        with DataLoader(data, rand_idx, self.device_train, num_threads=2, dim=1) as data_loader:
            for batch_index in range(num_batches):
                # prepare batch data
                batch = AttrDict(data_loader.get_next_batch())
                batch.states = self._augment_obs(batch.states)
                loss = self._impala_step(batch, self._do_log and batch_index == num_batches - 1)
                kls_target.append(batch.kl_target.mean().item())
                kls_replay.append(batch.kl_replay.mean().item())
                value_target_list.append(batch.state_value_targets.detach())

        kl_target = np.mean(kls_target)
        kl_replay = np.mean(kls_replay)

        if self._do_log:
            self.logger.add_scalar('Optimizer/Learning Rate', self._learning_rate, self.frame_train)
            if loss is not None:
                self.logger.add_scalar('Losses/Total Loss', loss, self.frame_train)
            self.logger.add_scalar('Stability/KL Blend', kl_target, self.frame_train)
            self.logger.add_scalar('Stability/KL Replay', kl_replay, self.frame_train)
            self.logger.add_scalar('Model Diff/Abs', model_diff(old_model, self._train_model), self.frame_train)
            self.logger.add_scalar('Model Diff/Max', model_diff(old_model, self._train_model, True), self.frame_train)

        self._unapply_pop_art(value_target_list)

        copy_state_dict(self._train_model, self._eval_model)

    def _apply_pop_art(self):
        if self.use_pop_art:
            self._train_model.heads.state_values.normalize(*self._pop_art.statistics)

    def _unapply_pop_art(self, value_target_list: List[Tensor]):
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

    def _add_reward_weights(self, data: AttrDict):
        horizon, num_rollouts = data.rewards.shape[:2]
        data.reward_weights = self._reward_weight_gen.generate(num_rollouts)\
            .unsqueeze(0).repeat(horizon, 1, 1)
        data.rewards = (data.rewards * self._reward_weight_gen.get_true_weights(data.reward_weights)).sum(-1)
        assert data.rewards.shape == (horizon, num_rollouts)

    def _augment_obs(self, states: Tensor) -> Tensor:
        if not self.random_crop_obs:
            return states
        assert states.ndim == 5
        if states.dtype == torch.uint8:
            states = states.float().div_(255)
        if self._crop is None:
            self._crop = kornia.augmentation.RandomCrop(states.shape[-2:], 4, padding_mode='replicate')
        return self._crop(states.view(-1, *states.shape[2:])).view_as(states)

    def _impala_step(self, batch, do_log):
        with torch.enable_grad():
            actor_params = AttrDict()
            if do_log:
                actor_params.logger = self.logger
                actor_params.cur_step = self.frame_train

            if self._train_model.is_recurrent:
                actor_out = self._run_train_model_rnn(batch.states, batch.memory, batch.dones, batch.reward_weights, batch.actions, actor_params)
            else:
                actor_out = self._run_train_model_feedforward(batch.states, batch.reward_weights, batch.actions, actor_params)
            batch.update(actor_out)

            for k, v in batch.items():
                batch[k] = v if k == 'states' or k.startswith('features') else v.cpu()

            if do_log:
                batch.state_values = log_gradients(batch.state_values, self.logger, 'Values', self.frame_train)
                batch.action_advantages = log_gradients(batch.action_advantages, self.logger, 'Q Values', self.frame_train)
                batch.dpg_values = log_gradients(batch.dpg_values, self.logger, 'Values Grad Act', self.frame_train)
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

        self._no_blend_batches += 1
        if self._no_blend_batches >= 10:
            self._no_blend_batches = 0
            lerp_module_(self._target_model, self._train_model, self.eval_model_blend)

        return loss

    def _run_train_model_feedforward(self, states, reward_weights, actions_replay, actor_params):
        ac_out_train = self._train_model(states, goal=reward_weights, evaluate_heads=['logits'], **actor_params)
        with torch.no_grad():
            ac_out_target = self._target_model(states, goal=reward_weights, evaluate_heads=['logits'])

        state_values = self._train_model.heads.state_values(ac_out_train.features_0)
        action_advantages = self._train_model.heads.action_values(ac_out_train.features_0, actions=actions_replay)

        actions_desired = self._train_model.heads.logits.pd.sample(ac_out_train.logits)
        dpg_values = self._calc_dpg(actions_desired, ac_out_train.features_0) \
            if self.dpg_loss_scale != 0 else state_values.detach()

        assert state_values.shape == dpg_values.shape == (*states.shape[:2], 1), \
            (state_values.shape, dpg_values.shape, (*states.shape[:2], 1), states.shape)

        return dict(
            logits=ac_out_train.logits,
            state_values=state_values.squeeze(-1),
            dpg_values=dpg_values.squeeze(-1),
            action_advantages=action_advantages.squeeze(-1),
            logits_target=ac_out_target.logits,
            **(dict(features_raw=ac_out_train.features_raw_0) if hasattr(ac_out_train, 'features_raw_0') else {}),
        )

    def _run_train_model_rnn(self, states, memory, dones, reward_weights, actions_replay, actor_params):
        with torch.no_grad():
            actor_out_burn_in = self._train_model(
                states[:self.memory_burn_in_steps], memory=memory[0], dones=dones[:self.memory_burn_in_steps],
                goal=reward_weights[:self.memory_burn_in_steps], evaluate_heads=['logits'])
            actor_out_target = self._target_model(states, memory=memory[0], dones=dones,
                                                  goal=reward_weights, evaluate_heads=['logits'])
        actor_out = self._train_model(
            states[self.memory_burn_in_steps:], memory=actor_out_burn_in.memory, dones=dones[self.memory_burn_in_steps:],
            goal=reward_weights[self.memory_burn_in_steps:], evaluate_heads=['logits'], **actor_params)

        logits = torch.cat([actor_out_burn_in.logits, actor_out.logits], 0)
        features = torch.cat([actor_out_burn_in.features_0, actor_out.features_0], 0)
        # features_raw = torch.cat([actor_out_burn_in.features_raw_0, actor_out.features_raw_0], 0)

        state_values = self._train_model.heads.state_values(features)
        action_advantages = self._train_model.heads.action_values(features, actions=actions_replay)

        actions_desired = self._train_model.heads.logits.pd.sample(logits)
        dpg_values = self._calc_dpg(actions_desired, features) \
            if self.dpg_loss_scale != 0 else state_values.detach()
        assert state_values.shape == dpg_values.shape == (*states.shape[:-1], 1)

        return dict(
            logits=logits,
            state_values=state_values.squeeze(-1),
            dpg_values=dpg_values.squeeze(-1),
            action_advantages=action_advantages.squeeze(-1),
            logits_target=actor_out_target.logits,
            # features_raw=features_raw,
        )

    def _calc_dpg(self, actions, features):
        assert actions.requires_grad

        head = self._train_model.heads.action_values

        for p in head.parameters():
            p.requires_grad = False

        action_advantages = head(features.detach(), actions=actions)

        for p in head.parameters():
            p.requires_grad = True

        assert action_advantages.requires_grad
        return action_advantages

    def _get_impala_loss(self, data, do_log=False, tag=''):
        pd = self._train_model.heads.logits.pd

        state_values, action_advantages = data.state_values, data.action_advantages
        data.update({k: v[:-1] for k, v in data.items()})
        data.state_values, data.action_advantages = state_values, action_advantages

        # action probability ratio
        data.logp_replay = pd.logp(data.actions, data.logits_replay)
        data.logp_target = pd.logp(data.actions, data.logits_target)
        data.logp = pd.logp(data.actions, data.logits)
        data.probs_ratio = (data.logp.detach() - data.logp_replay).exp()
        data.kl_replay = pd.kl(data.logits_replay, data.logits)

        with torch.no_grad():
            self._process_rewards(data, do_log)
        data.state_values = data.state_values[:-1]
        data.action_advantages = data.action_advantages[:-1]

        loss_world_model = self._get_world_model_loss(data, do_log)

        for k, v in data.items():
            data[k] = v.flatten(end_dim=1)

        data.kl_target = pd.kl(data.logits_target, data.logits)
        if LossType.v_mpo in self.loss_type:
            losses = v_mpo_loss(
                data.kl_target, data.logp, data.advantages, self.kl_pull)
            if losses is None:
                return None
            loss_target, loss_kl = losses
        elif LossType.impala in self.loss_type:
            losses = scaled_impala_loss(
                data.kl_target, data.logp, data.advantages, self.kl_pull, self.kl_limit)
            if losses is None:
                return None
            loss_target, loss_kl = losses

        entropy = pd.entropy(data.logits)
        loss_ent = self.entropy_loss_scale * -entropy.mean()
        loss_state_value = self.value_loss_scale * barron_loss(data.state_values, data.state_value_targets, *self.barron_alpha_c, reduce=False)
        loss_action_advantage = self.q_loss_scale * barron_loss(data.action_advantages, data.action_advantage_targets, *self.barron_alpha_c, reduce=False)

        # kl_mask = (data.kl_target.detach().mean(-1) <= self.kl_limit).float()
        # assert kl_mask.shape == data.dpg_values.shape
        loss_dpg = self.dpg_loss_scale * -data.dpg_values #* kl_mask

        assert loss_state_value.shape == data.state_values.shape, (loss_state_value.shape, data.state_values.shape)
        assert loss_action_advantage.shape == loss_dpg.shape == loss_state_value.shape

        loss_dpg = loss_dpg.mean()
        loss_ent = loss_ent.mean()
        loss_state_value = loss_state_value.mean()
        loss_action_advantage = loss_action_advantage.mean()

        # sum all losses
        total_loss = loss_target + loss_kl + loss_state_value + loss_action_advantage + loss_dpg + loss_world_model + loss_ent
        assert not np.isnan(total_loss.mean().item()) and not np.isinf(total_loss.mean().item()), \
            (loss_target.mean().item(), loss_state_value.mean().item())

        with torch.no_grad():
            if do_log:
                if self.use_pop_art:
                    pa_mean, pa_std = self._pop_art.statistics
                    data = AttrDict(**data)
                    data.state_values = data.state_values * pa_std + pa_mean
                    data.state_value_targets = data.state_value_targets * pa_std + pa_mean
                log_training_data(self._do_log, self.logger, self.frame_train, self._train_model, data)
                ratio = (data.logp - data.logp_target).exp() - 1
                self.logger.add_scalar('Prob Ratio/Mean' + tag, ratio.mean(), self.frame_train)
                self.logger.add_scalar('Prob Ratio/Abs Mean' + tag, ratio.abs().mean(), self.frame_train)
                self.logger.add_scalar('Prob Ratio/Abs Max' + tag, ratio.abs().max(), self.frame_train)
                self.logger.add_scalar('VTrace P/Mean', data.vtrace_p.mean(), self.frame_train)
                self.logger.add_scalar('VTrace P/Above 0.25 Fraction', (data.vtrace_p > 0.25).float().mean(), self.frame_train)
                self.logger.add_scalar('Stability/Entropy' + tag, entropy.mean(), self.frame_train)
                self.logger.add_scalar('Losses/State Value' + tag, loss_state_value.mean(), self.frame_train)
                self.logger.add_scalar('Losses/Q Value' + tag, loss_action_advantage.mean(), self.frame_train)
                if LossType.v_mpo is self.loss_type:
                    self.logger.add_scalar('Losses/KL Blend' + tag, loss_kl, self.frame_train)
                self.logger.add_histogram('Losses/Value Hist' + tag, loss_state_value, self.frame_train)
                self.logger.add_histogram('Losses/Ratio Hist' + tag, ratio, self.frame_train)

        return total_loss

    def _get_world_model_loss(self, data: AttrDict, do_log: bool) -> torch.Tensor:
        fx: FCFeatureExtractor = self._train_model.feature_extractors[0]
        if not hasattr(fx, 'run_world_model'):
            return 0

        data_values, data_rewards, data_logits, data_actions, data_dones = \
            [x.to(self.device_train) for x in (data.state_value_targets, data.rewards, data.logits, data.actions, data.dones)]

        sim_depth = fx.sim_depth
        assert data_rewards.ndim >= 2

        with torch.no_grad():
            h = data.states.shape[0]
            nondone = torch.ones_like(data_dones[:h - sim_depth])
            sliced_data = []
            for i in range(sim_depth):
                step_dones = data_dones[i: i + h - sim_depth]
                sliced_data.append([
                    data_rewards[i: i + h - sim_depth] * nondone,
                    data_actions[i: i + h - sim_depth],
                    data_values[i: i + h - sim_depth] * nondone,
                    data_logits[i: i + h - sim_depth],
                    (step_dones + (1 - nondone)).clamp_max(1),
                ])
                nondone *= 1 - step_dones
            rewards, actions, values, logits, dones = [torch.stack(x, 0).detach_() for x in zip(*sliced_data)]

        pred_values, pred_rewards, pred_logits, pred_dones = fx.run_world_model(data.features_raw[:h - sim_depth], actions)

        kl = barron_loss(pred_logits, logits, *self.barron_alpha_c, reduce=False)
        assert kl.shape[:-1] == dones.shape, (kl.shape, dones.shape)
        kl = ((1 - dones.unsqueeze(-1)) * kl).mean()

        value_loss = barron_loss(pred_values.squeeze(-1), values, *self.barron_alpha_c)
        reward_loss = 2 * barron_loss(pred_rewards[1:].squeeze(-1), rewards[:-1], *self.barron_alpha_c)
        done_loss = barron_loss(pred_dones[1:].squeeze(-1), dones[:-1], *self.barron_alpha_c)

        with torch.no_grad():
            if do_log:
                self.logger.add_scalar('Losses/World Value', value_loss, self.frame_train)
                self.logger.add_scalar('Losses/World KL', kl, self.frame_train)
                self.logger.add_scalar('Losses/World Reward', reward_loss, self.frame_train)
                self.logger.add_scalar('Losses/World Done', done_loss, self.frame_train)

        return (kl + value_loss + reward_loss + done_loss).cpu()

    def _process_rewards(self, data, do_log, mean_norm=True):
        if self.use_pop_art:
            pa_mean, pa_std = self._pop_art.statistics

        data.rewards = self.reward_scale * data.rewards

        state_values = data.state_values.detach() * pa_std + pa_mean if self.use_pop_art else data.state_values.detach()
        # calculate value targets and advantages
        state_value_targets, advantages, data.vtrace_p = calc_vtrace(
            data.rewards, state_values,
            data.dones, data.probs_ratio.detach().mean(-1), data.kl_replay.detach().mean(-1),
            self.reward_discount, self.vtrace_max_ratio, self.vtrace_kl_limit)

        action_values = data.action_advantages.detach() + state_values
        if self.use_pop_art:
            action_values = action_values * pa_std + pa_mean
        advantages_upgo = calc_upgo(data.rewards, state_values,
                                    data.dones, self.reward_discount,
                                    gae_lambda=1.0, action_values=action_values if self.q_loss_scale > 0 else None) - state_values[:-1]
        action_advantage_targets = data.rewards + self.reward_discount * (1 - data.dones) * state_value_targets[1:] - state_values[:-1]
        state_value_targets = state_value_targets[:-1]

        if do_log:
            self.logger.add_scalar('Advantages/Mean', advantages.mean(), self.frame_train)
            self.logger.add_scalar('Advantages/RMS', advantages.pow(2).mean().sqrt(), self.frame_train)
            self.logger.add_scalar('Advantages/UPGO Mean', advantages_upgo.mean(), self.frame_train)
            self.logger.add_scalar('Advantages/UPGO RMS', advantages_upgo.pow(2).mean().sqrt(), self.frame_train)
            self.logger.add_scalar('Advantages/Act Adv Mean', action_advantage_targets.mean(), self.frame_train)
            self.logger.add_scalar('Advantages/Act Adv RMS', action_advantage_targets.pow(2).mean().sqrt(), self.frame_train)
            self.logger.add_scalar('Advantages/Std', advantages.std(), self.frame_train)
            self.logger.add_scalar('Values/Values', state_values.mean(), self.frame_train)
            self.logger.add_scalar('Values/Values RMS', state_values.pow(2).mean().sqrt(), self.frame_train)
            self.logger.add_scalar('Values/Values Std', state_values.std(), self.frame_train)
            self.logger.add_scalar('Values/Value Targets', state_value_targets.mean(), self.frame_train)

        if self.use_pop_art:
            action_advantage_targets = action_advantage_targets / pa_std
            state_value_targets = (state_value_targets - pa_mean) / pa_std
            if LossType.impala is self.loss_type:
                advantages /= pa_std
                advantages_upgo /= pa_std
            if do_log:
                self.logger.add_scalar('Values/Values PopArt', data.state_values.mean(), self.frame_train)
                self.logger.add_scalar('Values/Value Targets PopArt', state_value_targets.mean(), self.frame_train)

        assert data.vtrace_p.shape == advantages_upgo.shape
        advantages = self._adv_norm(self.pg_loss_scale * advantages + self.upgo_scale * data.vtrace_p * advantages_upgo)

        data.state_value_targets, data.advantages = state_value_targets, advantages
        data.action_advantage_targets = action_advantage_targets

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