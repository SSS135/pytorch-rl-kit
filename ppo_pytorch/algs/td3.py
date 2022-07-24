import math
from asyncio import Future
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from functools import partial
from typing import Optional

import gym.spaces
import torch
import torch.autograd
import torch.autograd
import torch.nn.functional as F
import torch.optim as optim
from ppo_pytorch.actors.actors import Actor
from ppo_pytorch.actors.fc_actors import create_sac_fc_actor
from ppo_pytorch.algs.ppo import copy_state_dict
from ppo_pytorch.common.gae import calc_vtrace
from torch.nn.utils import clip_grad_norm_
from torchvision.utils import make_grid

from .replay_buffer import ReplayBuffer
from .utils import lerp_module_
from ..actors.actors import ModularActor
from ..actors.utils import model_diff
from ..common.attr_dict import AttrDict
from ..common.barron_loss import barron_loss
from ..common.data_loader import DataLoader
from ..common.probability_distributions import DiagGaussianPd, CategoricalPd, FixedStdGaussianPd
from ..common.rl_base import RLBase, RLStepData


class TD3(RLBase):
    def __init__(self, observation_space, action_space,
                 reward_discount=0.99,
                 train_interval=16,
                 batch_size=128,
                 num_batches=16,
                 rollout_length=2,
                 replay_buffer_size=128*1024,
                 replay_end_sampling_factor=1.0,
                 target_model_blend=0.005,
                 actor_update_interval=2,
                 random_policy_frames=10000,
                 entropy_scale=0.2,
                 kl_pull=0.5,
                 model_factory=create_sac_fc_actor,
                 actor_optimizer_factory=partial(optim.Adam, lr=5e-4),
                 critic_optimizer_factory=partial(optim.Adam, lr=5e-4),
                 cuda_eval=False,
                 cuda_train=False,
                 grad_clip_norm=None,
                 reward_scale=1.0,
                 lr_scheduler_factory=None,
                 entropy_decay_factory=None,
                 vtrace_kl_limit=0.2,
                 **kwargs):
        super().__init__(observation_space, action_space, **kwargs)
        self._init_args = locals()
        self.reward_discount = reward_discount
        self.train_interval = train_interval
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.rollout_length = rollout_length
        self.replay_buffer_size = replay_buffer_size
        self.replay_end_sampling_factor = replay_end_sampling_factor
        self.target_model_blend = target_model_blend
        self.actor_update_interval = actor_update_interval
        self.random_policy_frames = random_policy_frames
        self.entropy_scale = entropy_scale
        self.kl_pull = kl_pull
        self.model_factory = model_factory
        self.device_eval = torch.device('cuda' if cuda_eval else 'cpu')
        self.device_train = torch.device('cuda' if cuda_train else 'cpu')
        self.grad_clip_norm = grad_clip_norm
        self.reward_scale = reward_scale
        self.vtrace_kl_limit = vtrace_kl_limit

        assert self.num_rewards == 1
        assert rollout_length >= 2
        assert batch_size % rollout_length == 0
        assert isinstance(action_space, gym.spaces.Box), action_space

        self._train_model: ModularActor = model_factory(observation_space, action_space)
        self._eval_model: ModularActor = model_factory(observation_space, action_space)
        self._target_model: ModularActor = model_factory(observation_space, action_space)
        if self.model_init_path is not None:
            self._train_model.load_state_dict(torch.load(self.model_init_path), True)
            print(f'loaded model {self.model_init_path}')
        copy_state_dict(self._train_model, self._eval_model)
        copy_state_dict(self._train_model, self._target_model)
        self._train_model = self._train_model.train().to(self.device_train, non_blocking=True)
        self._train_model_copy = deepcopy(self._train_model)
        self._eval_model = self._eval_model.eval().to(self.device_eval, non_blocking=True)
        self._target_model = self._target_model.train().to(self.device_train, non_blocking=True)

        assert isinstance(self._eval_model.heads.logits.pd, FixedStdGaussianPd)

        self._actor_optimizer: torch.optim.Optimizer = \
            actor_optimizer_factory(self._train_model.head_parameters('logits'))
        self._critic_optimizer: torch.optim.Optimizer = \
            critic_optimizer_factory(self._train_model.head_parameters('q1', 'q2'))
        self._actor_lr_scheduler = lr_scheduler_factory(self._actor_optimizer) if lr_scheduler_factory is not None else None
        self._critic_lr_scheduler = lr_scheduler_factory(self._critic_optimizer) if lr_scheduler_factory is not None else None
        self._entropy_decay = entropy_decay_factory() if entropy_decay_factory is not None else None
        self._replay_buffer = ReplayBuffer(replay_buffer_size, replay_end_sampling_factor)
        self._eval_steps = 0
        self._prev_data = None
        self._last_model_save_frame = 0
        self._train_future: Optional[Future] = None
        self._update_iter = 0

    def _step(self, data: RLStepData) -> torch.Tensor:
        with torch.no_grad():
            ac_out = self._eval_model(data.obs.to(self.device_eval), evaluate_heads=['logits'])
            # pd = self._eval_model.heads.logits.pd
            actions = (ac_out.logits + 0.3 * torch.randn_like(ac_out.logits)).cpu().clamp(-1, 1)
            if self.frame_eval < self.random_policy_frames:
                actions.uniform_(-1, 1)
                # actions = 0.5 * torch.log((1 + actions) / (1 - actions))

            if not self.disable_training:
                if self._prev_data is not None and data.rewards is not None:
                    self._replay_buffer.push(rewards=data.rewards, dones=data.done, **self._prev_data)

                self._eval_steps += 1
                self._prev_data = dict(logits=ac_out.logits, states=data.obs, actions=actions)

                if self.frame_eval > self.random_policy_frames \
                        and self._eval_steps >= self.train_interval:
                    self._eval_steps = 0
                    self._pre_train()
                    self._train()

            return actions

    def _pre_train(self):
        self.frame_train = self.frame_eval

        self._check_log()

        # update clipping and learning rate decay schedulers
        if self._actor_lr_scheduler is not None:
            self._actor_lr_scheduler.step(self.frame_train)
        if self._critic_lr_scheduler is not None:
            self._critic_lr_scheduler.step(self.frame_train)
        if self._entropy_decay is not None:
            self._entropy_decay.step(self.frame_train)

    def _train(self):
        data = self._create_data()
        self._do_train(data)

    def _do_train(self, data):
        with torch.no_grad():
            self._sac_update(data)
            self._model_saver.check_save_model(self._train_model, self.frame_train)

    def _create_data(self):
        # (steps, actors, *)
        data = self._replay_buffer.sample(self.batch_size // self.rollout_length * self.num_batches, self.rollout_length)
        data = AttrDict(data)
        data.rewards = self.reward_scale * data.rewards
        return data

    def _sac_update(self, data: AttrDict):
        num_rollouts = data.states.shape[1]

        data = AttrDict(states=data.states, logits_old=data.logits, actions=data.actions,
                        rewards=data.rewards[..., 0], dones=data.dones)

        rand_idx = torch.randperm(num_rollouts, device=self.device_train).chunk(self.num_batches)

        old_model = {k: v.clone() for k, v in self._train_model.state_dict().items()}

        with DataLoader(data, rand_idx, self.device_train, 2, dim=1) as data_loader:
            for batch_index in range(self.num_batches):
                # prepare batch data
                batch = AttrDict(data_loader.get_next_batch())
                loss = self._batch_update(batch, self._do_log and batch_index == self.num_batches - 1)
                if self._do_actor_update:
                    lerp_module_(self._target_model, self._train_model, self.target_model_blend)
                self._update_iter += 1

        if self._do_log:
            self.logger.add_scalar('Optimizer/Learning Rate', self._actor_optimizer.param_groups[0]['lr'], self.frame_train)
            if loss is not None:
                self.logger.add_scalar('Losses/Total Loss', loss, self.frame_train)
            self.logger.add_scalar('Model Diff/Abs', model_diff(old_model, self._train_model), self.frame_train)
            self.logger.add_scalar('Model Diff/Max', model_diff(old_model, self._train_model, True), self.frame_train)

        lerp_module_(self._eval_model, self._train_model, 1)

    @property
    def _do_actor_update(self):
        return self._update_iter % self.actor_update_interval == self.actor_update_interval - 1

    def _batch_update(self, batch, do_log=False):
        self._train_model.zero_grad()
        critc_loss = self._critic_step(batch, do_log)
        critc_loss.backward()
        if self.grad_clip_norm is not None:
            clip_grad_norm_(self._train_model.parameters(), self.grad_clip_norm)
        self._critic_optimizer.step()

        if self._do_actor_update:
            self._train_model.zero_grad()
            self._train_model_copy.zero_grad()
            lerp_module_(self._train_model_copy, self._train_model, 1)
            actor_loss = self._actor_step(batch, do_log)
            actor_loss.backward()
            if self.grad_clip_norm is not None:
                clip_grad_norm_(self._train_model.parameters(), self.grad_clip_norm)
            self._actor_optimizer.step()
        else:
            actor_loss = 0

        return actor_loss + critc_loss

    def _critic_step(self, data: AttrDict, do_log):
        pd = self._train_model.heads.logits.pd

        logits = self._target_model(data.states, evaluate_heads=['logits']).logits
        actions = logits  # pd.sample(logits)
        logp = pd.logp(actions, logits)

        ac_out_target = self._target_model(data.states, evaluate_heads=['q1', 'q2'], actions=actions)
        assert ac_out_target.q1.shape == (*logp.shape[:-1], 1)
        q_values = torch.min(ac_out_target.q1, ac_out_target.q2).squeeze(-1) - self.entropy_scale * logp.mean(-1)

        logp_ratio = (pd.logp(data.actions, logits) - pd.logp(data.actions, data.logits_old)).mean(-1)
        probs_ratio = logp_ratio.exp()
        kl_replay = pd.kl(data.logits_old, logits).mean(-1)
        vtrace_targets, _, vtrace_c = calc_vtrace(
            data.rewards[:-1], q_values,
            data.dones[:-1], probs_ratio[:-1], logp_ratio[:-1],
            self.reward_discount, self.vtrace_kl_limit, 1.0)
        targets = data.rewards[:-1] + self.reward_discount * (1 - data.dones[:-1]) * vtrace_targets[1:]
        assert data.rewards.shape == data.dones.shape == q_values.shape
        # targets = data.rewards[:-1] + self.reward_discount * (1 - data.dones[:-1]) * q_values[1:]

        assert (targets.shape[0] + 1, *targets.shape[1:]) == data.rewards.shape == logp.shape[:-1], (targets.shape, data.rewards.shape, logp.shape)

        with torch.enable_grad():
            ac = data.logits_old[:-1]
            ac = (ac + torch.empty_like(ac).uniform_(-0.1, 0.1)).clamp(-1, 1)
            ac_out_first = self._train_model(data.states[:-1], evaluate_heads=['q1', 'q2'], actions=ac)
            q1_pred = ac_out_first.q1.squeeze(-1)
            q2_pred = ac_out_first.q2.squeeze(-1)
            assert q1_pred.shape == q2_pred.shape == targets.shape
            loss = 0.5 * (q1_pred - targets).pow(2).mean() + 0.5 * (q2_pred - targets).pow(2).mean()

        if do_log:
            q1, q2 = ac_out_first.q1, ac_out_first.q2
            q_targ_1step = data.rewards[:-1] + self.reward_discount * (1 - data.dones[:-1]) * q_values[1:]
            self.logger.add_scalar('Stability/Entropy', pd.entropy(logits).mean(), self.frame_train)
            self.logger.add_scalar('Losses/State Value', loss, self.frame_train)
            self.logger.add_scalar('Values/Q1', q1.mean(), self.frame_train)
            q12_min = torch.min(q1, q2)
            q12t1t2_min = torch.stack([q1, q2, ac_out_target.q1[:-1], ac_out_target.q2[:-1]], 0).min(0)[0]
            self.logger.add_scalar('Values/Q12 Min', q12_min.mean(), self.frame_train)
            self.logger.add_scalar('Values/Q12T1T2 Min', q12t1t2_min.mean(), self.frame_train)
            self.logger.add_scalar('Values/Q1 - Q12 Min', (q1 - q12_min).mean(), self.frame_train)
            self.logger.add_scalar('Values/Q1 - Q12T1T2 Min', (q1 - q12t1t2_min).mean(), self.frame_train)
            self.logger.add_scalar('Value Errors/RMSE 1 Step', (q1.squeeze(-1) - q_targ_1step).pow(2).mean().sqrt(), self.frame_train)
            self.logger.add_scalar('Value Errors/RMSE', (q1.squeeze(-1) - targets).pow(2).mean().sqrt(), self.frame_train)
            self.logger.add_scalar('Value Errors/Abs', (q1.squeeze(-1) - targets).abs().mean(), self.frame_train)
            self.logger.add_scalar('Value Errors/Max', (q1.squeeze(-1) - targets).abs().max(), self.frame_train)
            self.logger.add_scalar('Stability/VTrace Mean', vtrace_c.mean(), self.frame_train)
            self.logger.add_scalar('Stability/KL Replay', kl_replay.mean(), self.frame_train)
            self.logger.add_scalar('Stability/LogP Ratio Mean', logp_ratio.mean(), self.frame_train)
            self.logger.add_scalar('Stability/LogP Ratio AbsMean', logp_ratio.abs().mean(), self.frame_train)
            self.logger.add_scalar('Stability/Logits Old Diff RMS', (logits - data.logits_old).pow(2).mean().sqrt(), self.frame_train)

        return loss

    def _actor_step(self, data: AttrDict, do_log):
        pd = self._train_model.heads.logits.pd
        logits_target = self._target_model(data.states, evaluate_heads=['logits']).logits

        with torch.enable_grad():
            logits = self._train_model(data.states, evaluate_heads=['logits']).logits
            actions = logits  # pd.sample(logits)
            logp = pd.logp(actions, logits).mean(-1)

            q1 = self._train_model_copy(data.states, evaluate_heads=['q1'], actions=actions).q1.squeeze(-1)
            pull_loss = self.kl_pull * 0.5 * (logits_target - logits).pow(2).mean()
            bounds_loss = 0.5 * (logits.abs().clamp_min(0.95) - 0.95).pow(2).mean()
            ent_loss = self.entropy_scale * logp.mean()
            q_loss = -q1.mean()
            loss = q_loss + ent_loss + pull_loss + bounds_loss
            # logits_grad = torch.autograd.grad(-q1.sum(), logits)[0].detach()
            # assert logits_grad.shape == logits.shape
            # logits_grad[(logits > 1) & (logits_grad < 0)] = 0
            # logits_grad[(logits < -1) & (logits_grad > 0)] = 0
            # pull_loss = self.kl_pull * 0.5 * (logits_target - logits).pow(2).mean()
            # # bounds_loss = logits.abs().clamp_min(1.0).mean()
            # ent_loss = self.entropy_scale * logp.mean()
            # q_loss = (logits_grad * logits).mean() #-q1.mean()
            # loss = q_loss + ent_loss + pull_loss #+ bounds_loss
            assert data.rewards.shape == q1.shape, (logp.shape, data.rewards.shape, q1.shape, loss.shape)

        if do_log:
            self.logger.add_scalar('Stability/KL Target', pd.kl(logits_target, logits).mean(), self.frame_train)
            self.logger.add_scalar('Stability/Logits Target Diff RMS', (logits - logits_target).pow(2).mean().sqrt(), self.frame_train)
            self.logger.add_scalar('Stability/OOB Actions', (logits.abs() > 1).float().mean(), self.frame_train)

        return loss

    def drop_collected_steps(self):
        self._prev_data = None