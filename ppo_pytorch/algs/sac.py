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
from ppo_pytorch.actors import Actor
from ppo_pytorch.actors.fc_actors import create_sac_fc_actor
from ppo_pytorch.algs.ppo import copy_state_dict
from ppo_pytorch.common.gae import calc_vtrace
from torch.nn.utils import clip_grad_norm_
from torchvision.utils import make_grid

from .replay_buffer import ReplayBuffer
from .utils import lerp_module_
from ..actors import ModularActor
from ..actors.utils import model_diff
from ..common.attr_dict import AttrDict
from ..common.barron_loss import barron_loss
from ..common.data_loader import DataLoader
from ..common.probability_distributions import DiagGaussianPd, CategoricalPd
from ..common.rl_base import RLBase, RLStepData


class SAC(RLBase):
    def __init__(self, observation_space, action_space,
                 reward_discount=0.99,
                 train_interval=16,
                 batch_size=128,
                 num_batches=16,
                 rollout_length=2,
                 replay_buffer_size=128*1024,
                 target_model_blend=0.005,
                 actor_update_interval=2,
                 random_policy_frames=10000,
                 entropy_scale=0.2,
                 kl_pull=0.05,
                 model_factory=create_sac_fc_actor,
                 actor_optimizer_factory=partial(optim.Adam, lr=5e-4),
                 critic_optimizer_factory=partial(optim.Adam, lr=5e-4),
                 cuda_eval=False,
                 cuda_train=False,
                 grad_clip_norm=None,
                 reward_scale=1.0,
                 # barron_alpha_c=(1.5, 1),
                 lr_scheduler_factory=None,
                 entropy_decay_factory=None,
                 # use_pop_art=False,
                 vtrace_max_ratio=1.0,
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
        # self.barron_alpha_c = barron_alpha_c
        # self.use_pop_art = use_pop_art
        self.vtrace_max_ratio = vtrace_max_ratio
        self.vtrace_kl_limit = vtrace_kl_limit

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
        self._eval_model = self._eval_model.eval().to(self.device_eval, non_blocking=True)
        self._target_model = self._target_model.train().to(self.device_train, non_blocking=True)

        assert isinstance(self._eval_model.heads.logits.pd, DiagGaussianPd)

        self._actor_optimizer: torch.optim.Optimizer = \
            actor_optimizer_factory(self._train_model.head_parameters('logits'))
        self._critic_optimizer: torch.optim.Optimizer = \
            critic_optimizer_factory(self._train_model.head_parameters('q1', 'q2'))
        self._actor_lr_scheduler = lr_scheduler_factory(self._actor_optimizer) if lr_scheduler_factory is not None else None
        self._critic_lr_scheduler = lr_scheduler_factory(self._critic_optimizer) if lr_scheduler_factory is not None else None
        self._entropy_decay = entropy_decay_factory() if entropy_decay_factory is not None else None
        self._replay_buffer = ReplayBuffer(replay_buffer_size)
        # self._pop_art = PopArt()
        # self._train_executor = ThreadPoolExecutor(max_workers=1)
        self._eval_steps = 0
        self._prev_data = None
        self._last_model_save_frame = 0
        self._train_future: Optional[Future] = None
        self._update_iter = 0

    def _step(self, data: RLStepData) -> torch.Tensor:
        with torch.no_grad():
            ac_out = self._eval_model(data.obs.to(self.device_eval), evaluate_heads=['logits'])
            pd = self._eval_model.heads.logits.pd
            actions = pd.sample(ac_out.logits).cpu()
            if self.frame_eval < self.random_policy_frames:
                actions.uniform_(-0.97, 0.97)
                actions = 0.5 * torch.log((1 + actions) / (1 - actions))

            if not self.disable_training:
                if self._prev_data is not None and data.rewards is not None:
                    self._replay_buffer.push(rewards=data.rewards, dones=data.done, **self._prev_data)

                self._eval_steps += 1
                self._prev_data = dict(logits=ac_out.logits, states=data.obs, actions=actions)

                min_replay_size = self.batch_size * self.rollout_length
                if self.frame_eval > self.random_policy_frames \
                        and self._eval_steps >= self.train_interval \
                        and len(self._replay_buffer) >= min_replay_size:
                    self._eval_steps = 0
                    self._pre_train()
                    self._train()

            return actions.tanh()

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
        # if self._train_future is not None:
        #     self._train_future.result()
        # self._train_future = self._train_executor.submit(self._train_async, data)

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
            # self.logger.add_scalar('Stability/KL Blend', kl_policy, self.frame_train)
            # self.logger.add_scalar('Stability/KL Replay', kl_replay, self.frame_train)
            self.logger.add_scalar('Model Diff/Abs', model_diff(old_model, self._train_model), self.frame_train)
            self.logger.add_scalar('Model Diff/Max', model_diff(old_model, self._train_model, True), self.frame_train)

        lerp_module_(self._eval_model, self._train_model, 1)

    @property
    def _do_actor_update(self):
        return self._update_iter % self.actor_update_interval == 0

    def _batch_update(self, batch, do_log=False):
        self._train_model.zero_grad()
        critc_loss = self._critic_step(batch, do_log)
        critc_loss.backward()
        if self.grad_clip_norm is not None:
            clip_grad_norm_(self._train_model.parameters(), self.grad_clip_norm)
        self._critic_optimizer.step()

        if self._do_actor_update:
            self._train_model.zero_grad()
            actor_loss = self._actor_step(batch, do_log)
            actor_loss.backward()
            if self.grad_clip_norm is not None:
                clip_grad_norm_(self._train_model.parameters(), self.grad_clip_norm)
            self._actor_optimizer.step()
        else:
            actor_loss = 0

        # with torch.no_grad():
        #     if do_log:
        #         self._log_training_data(data)
        #         ratio = (data.logp - data.logp_policy).exp() - 1
        #         self.logger.add_scalar('Prob Ratio/Mean' + tag, ratio.mean(), self.frame_train)
        #         self.logger.add_scalar('Prob Ratio/Abs Mean' + tag, ratio.abs().mean(), self.frame_train)
        #         self.logger.add_scalar('Prob Ratio/Abs Max' + tag, ratio.abs().max(), self.frame_train)
        #         self.logger.add_scalar('Stability/Entropy' + tag, entropy.mean(), self.frame_train)
        #         self.logger.add_scalar('Losses/State Value' + tag, loss_value.mean(), self.frame_train)
        #         self.logger.add_histogram('Losses/Value Hist' + tag, loss_value, self.frame_train)
        #         self.logger.add_histogram('Losses/Ratio Hist' + tag, ratio, self.frame_train)

        return actor_loss + critc_loss

    def _critic_step(self, data: AttrDict, do_log):
        pd = self._train_model.heads.logits.pd

        logits = self._train_model(data.states, evaluate_heads=['logits']).logits
        actions = pd.sample(logits)
        logp = pd.logp(actions, logits)

        ac_out = self._target_model(data.states, evaluate_heads=['q1', 'q2'], actions=actions.tanh(), logits=logits)
        q_values = torch.min(ac_out.q1, ac_out.q2).squeeze(-1) - self.entropy_scale * logp.mean(-1)

        probs_ratio = (pd.logp(data.actions, logits) - pd.logp(data.actions, data.logits_old)).exp()
        kl_replay = pd.kl(data.logits_old, logits)

        vtrace_targets, _, _, _ = calc_vtrace(
            data.rewards[1:-1], q_values[1:],
            data.dones[1:-1], probs_ratio[1:-1].detach().mean(-1), kl_replay[1:-1].detach().mean(-1),
            self.reward_discount, self.vtrace_max_ratio, self.vtrace_kl_limit)
        targets = data.rewards[:-2] + self.reward_discount * (1 - data.dones[:-2]) * vtrace_targets

        # for i in reversed(range(self.value_target_steps)):
        #     targets = data.rewards[i] + self.reward_discount * (1 - data.dones[i]) * targets
        # assert self.value_target_steps == 1 and data.states.shape[0] == 2
        # targets = data.rewards[0] + self.reward_discount * (1 - data.dones[0]) * targets

        assert (targets.shape[0] + 2, *targets.shape[1:]) == data.rewards.shape == logp.shape[:-1], (targets.shape, data.rewards.shape, logp.shape)

        with torch.enable_grad():
            ac = data.actions[:-2].tanh()
            # ac = ac + torch.empty_like(ac).uniform_(-0.1, 0.1)
            ac_out_first = self._train_model(data.states[:-2], evaluate_heads=['q1', 'q2'], actions=ac, logits=data.logits_old[:-2])
            loss = (ac_out_first.q1.squeeze(-1) - targets) ** 2 + (ac_out_first.q2.squeeze(-1) - targets) ** 2
            assert targets.shape == loss.shape
            loss = loss.mean()

        return loss

    def _actor_step(self, data: AttrDict, do_log):
        pd = self._train_model.heads.logits.pd
        logits_target = self._target_model(data.states, evaluate_heads=['logits']).logits

        with torch.enable_grad():
            logits = self._train_model(data.states, evaluate_heads=['logits']).logits
            actions = pd.sample(logits)
            logp = pd.logp(actions, logits).mean(-1)

            ac_out = self._train_model(data.states, evaluate_heads=['q1', 'q2'], actions=actions.tanh(), logits=logits)
            q_target = torch.min(ac_out.q1, ac_out.q2).squeeze(-1)
            kl = pd.kl(logits_target, logits)
            loss = self.entropy_scale * logp - q_target + self.kl_pull * kl.mean(-1)
            assert logp.shape == data.rewards.shape == q_target.shape == loss.shape, (logp.shape, data.rewards.shape, q_target.shape, loss.shape)
            return loss.mean()

    def drop_collected_steps(self):
        self._prev_data = None

    def _log_training_data(self, data: AttrDict):
        if self._do_log:
            if data.states.dim() == 4:
                if data.states.shape[1] in (1, 3):
                    img = data.states[:4]
                    nrow = 2
                else:
                    img = data.states[:4]
                    img = img.view(-1, *img.shape[2:]).unsqueeze(1)
                    nrow = data.states.shape[1]
                if data.states.dtype == torch.uint8:
                    img = img.float() / 255
                img = make_grid(img, nrow=nrow, normalize=False)
                self.logger.add_image('state', img, self.frame_train)
            # vsize = data.value_targets.shape[-2] ** 0.5
            # targets = data.value_targets.sum(-2) / vsize
            # values = data.state_values.sum(-2) / vsize
            # v_mean = values.mean(-1)
            # t_mean = targets.mean(-1)
            self.logger.add_histogram('rewards', data.rewards, self.frame_train)
            # self.logger.add_histogram('value_targets', targets, self.frame)
            # self.logger.add_histogram('advantages', data.advantages, self.frame)
            # self.logger.add_histogram('values', values, self.frame)
            # self.logger.add_scalar('value rmse', (v_mean - t_mean).pow(2).mean().sqrt(), self.frame)
            # self.logger.add_scalar('value abs err', (v_mean - t_mean).abs().mean(), self.frame)
            # self.logger.add_scalar('value max err', (v_mean - t_mean).abs().max(), self.frame)
            if isinstance(self._train_model.heads.logits.pd, DiagGaussianPd):
                mean, std = data.logits.chunk(2, dim=1)
                self.logger.add_histogram('logits mean', mean, self.frame_train)
                self.logger.add_histogram('logits std', std, self.frame_train)
            elif isinstance(self._train_model.heads.logits.pd, CategoricalPd):
                self.logger.add_histogram('logits log_softmax', F.log_softmax(data.logits, dim=-1), self.frame_train)
            self.logger.add_histogram('logits', data.logits, self.frame_train)
            for name, param in self._train_model.named_parameters():
                self.logger.add_histogram(name, param, self.frame_train)