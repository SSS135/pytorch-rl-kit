from asyncio import Future
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from functools import partial
from typing import Optional

import gym.spaces
import math
import torch
import torch.autograd
import torch.autograd
import torch.nn.functional as F
import torch.optim as optim
from ..common.pop_art import PopArt
from torch.nn.utils import clip_grad_norm_
from torchvision.utils import make_grid

from .replay_buffer import ReplayBuffer
from .utils import blend_models
from ..actors import ModularActor, create_td3_fc_actor
from ..actors.utils import model_diff
from ..common.attr_dict import AttrDict
from ..common.barron_loss import barron_loss
from ..common.data_loader import DataLoader
from ..common.probability_distributions import DiagGaussianPd, CategoricalPd
from ..common.rl_base import RLBase
from ..common.gae import calc_value_targets
from rl_exp.noisy_linear import NoisyLinear


class ODDPG(RLBase):
    def __init__(self, observation_space, action_space,
                 reward_discount=0.99,
                 horizon=64,
                 critic_batch_size=128,
                 actor_batch_size=128,
                 num_actor_batches=16,
                 critic_iters=4,
                 value_clip=0.2,
                 replay_buffer_size=128 * 1024,
                 target_model_blend=0.05,
                 expl_noise_scale=0.1,
                 random_policy_frames=20000,
                 model_factory=create_td3_fc_actor,
                 actor_optimizer_factory=partial(optim.Adam, lr=3e-4),
                 critic_optimizer_factory=partial(optim.Adam, lr=3e-4),
                 cuda_eval=True,
                 cuda_train=True,
                 grad_clip_norm=None,
                 reward_scale=1.0,
                 barron_alpha_c=(1.5, 1),
                 lr_scheduler_factory=None,
                 entropy_decay_factory=None,
                 use_pop_art=False,
                 **kwargs):
        super().__init__(observation_space, action_space, **kwargs)
        self._init_args = locals()
        self.reward_discount = reward_discount
        self.horizon = horizon
        self.critic_batch_size = critic_batch_size
        self.actor_batch_size = actor_batch_size
        self.num_actor_batches = num_actor_batches
        self.replay_buffer_size = replay_buffer_size
        self.target_model_blend = target_model_blend
        self.expl_noise_scale = expl_noise_scale
        self.critic_iters = critic_iters
        self.random_policy_frames = random_policy_frames
        self.model_factory = model_factory
        self.device_eval = torch.device('cuda' if cuda_eval else 'cpu')
        self.device_train = torch.device('cuda' if cuda_train else 'cpu')
        self.grad_clip_norm = grad_clip_norm
        self.reward_scale = reward_scale
        self.barron_alpha_c = barron_alpha_c
        self.value_clip = value_clip
        self.use_pop_art = use_pop_art

        assert isinstance(action_space, gym.spaces.Box), action_space

        self._train_model: ModularActor = model_factory(observation_space, action_space)
        if self.model_init_path is not None:
            self._train_model.load_state_dict(torch.load(self.model_init_path))
            print(f'loaded model {self.model_init_path}')

        self._train_model = self._train_model.to(self.device_train).train()
        self._target_model = deepcopy(self._train_model)
        self._eval_model = deepcopy(self._train_model).to(self.device_eval).eval()

        self._actor_optimizer: torch.optim.Optimizer = \
            actor_optimizer_factory(self._train_model.head_parameters('logits'))
        self._critic_optimizer: torch.optim.Optimizer = \
            critic_optimizer_factory(self._train_model.head_parameters('state_values_1', 'state_values_2'))
        self._actor_lr_scheduler = lr_scheduler_factory(self._actor_optimizer) if lr_scheduler_factory is not None else None
        self._critic_lr_scheduler = lr_scheduler_factory(self._critic_optimizer) if lr_scheduler_factory is not None else None
        self._entropy_decay = entropy_decay_factory() if entropy_decay_factory is not None else None
        self._replay_buffer = ReplayBuffer(replay_buffer_size)
        self._pop_art = PopArt()
        self._eval_steps = 0
        self._prev_data = None

    def _step(self, rewards, dones, states) -> torch.Tensor:
        with torch.no_grad():
            # run network
            ac_out = self._eval_model(states.to(self.device_eval), evaluate_heads=['logits'])
            noise = self.expl_noise_scale * torch.randn(ac_out.logits.shape) if not self.disable_training else 0
            pd = self._eval_model.heads.logits.pd
            actions = (pd.sample(ac_out.logits).cpu() + noise).clamp(-pd.max_action, pd.max_action)

            if not self.disable_training:
                if self._prev_data is not None and rewards is not None:
                    self._replay_buffer.push(rewards=rewards, dones=dones, **self._prev_data)

                self._eval_steps += 1
                self._prev_data = dict(logits=ac_out.logits, states=states, actions=actions)

                if self._eval_steps > self.horizon + 1:
                    self._eval_steps = 0
                    self._check_log()
                    self._train()
                    self._scheduler_step()

            return actions

    def _scheduler_step(self):
        # update clipping and learning rate decay schedulers
        if self._actor_lr_scheduler is not None:
            self._actor_lr_scheduler.step(self.frame)
        if self._critic_lr_scheduler is not None:
            self._critic_lr_scheduler.step(self.frame)
        if self._entropy_decay is not None:
            self._entropy_decay.step(self.frame)

    def _train(self):
        with torch.no_grad():
            # (steps, actors, *)
            actor_data = AttrDict(self._replay_buffer.sample(self.actor_batch_size * self.num_actor_batches, 1))
            critic_data = AttrDict(self._replay_buffer.get_last_samples(self.horizon + 1))
            self._oddpg_update(actor_data, critic_data)
            self._model_saver.check_save_model(self._train_model, self.frame)

    def _oddpg_update(self, actor_data: AttrDict, critic_data: AttrDict):
        old_model = deepcopy(self._train_model)

        critic_loss, critic_data = self._critic_update(critic_data)
        actor_loss = self._actor_update(actor_data) if self.frame > self.random_policy_frames else 0
        NoisyLinear.randomize_network(self._train_model)

        blend_models(self._train_model, self._target_model, self.target_model_blend)
        self._eval_model = deepcopy(self._train_model).to(self.device_eval).eval()

        if self._do_log:
            self.logger.add_scalar('learning rate', self._actor_optimizer.param_groups[0]['lr'], self.frame)
            self.logger.add_scalar('actor loss', actor_loss, self.frame)
            self.logger.add_scalar('critic loss', critic_loss, self.frame)
            self.logger.add_scalar('model abs diff', model_diff(old_model, self._train_model), self.frame)
            self.logger.add_scalar('model max diff', model_diff(old_model, self._train_model, True), self.frame)
            self._log_training_data(critic_data)

    def _critic_update(self, data: AttrDict):
        data.rewards = self.reward_scale * data.rewards

        self._calc_q_values(data)
        values_old = data.values_old
        data = AttrDict({k: v[:-1] for k, v in data.items()})
        data.value_targets = calc_value_targets(data.rewards, values_old, data.dones, self.reward_discount)

        data = AttrDict({k: v.flatten(end_dim=1) for k, v in data.items()})

        num_samples = data.states.shape[0]
        num_batches = max(1, math.ceil(num_samples / self.critic_batch_size))
        rand_idx = torch.randperm(num_samples * self.critic_iters, device=self.device_train)
        rand_idx = rand_idx.fmod_(num_samples).chunk(num_batches * self.critic_iters)

        with DataLoader(data, rand_idx, self.device_train, 4, dim=0) as data_loader:
            for batch_index in range(self.critic_iters * num_batches):
                # prepare batch data
                batch = AttrDict(data_loader.get_next_batch())
                loss = self._train_critic_on_batch(batch)

        return loss, data

    def _actor_update(self, data: AttrDict):
        data = dict(states=data.states)
        data = AttrDict({k: v.flatten(end_dim=1) for k, v in data.items()})

        num_samples = data.states.shape[0]
        num_batches = max(1, math.ceil(num_samples / self.actor_batch_size))
        rand_idx = torch.randperm(num_samples, device=self.device_train).chunk(num_batches)

        with DataLoader(data, rand_idx, self.device_train, 4, dim=0) as data_loader:
            for batch_index in range(num_batches):
                # prepare batch data
                batch = AttrDict(data_loader.get_next_batch())
                loss = self._train_actor_on_batch(batch)

        return loss

    def _calc_q_values(self, data: AttrDict):
        horizon, num_actors, *_ = data.states.shape
        num_batches = max(1, math.ceil(num_actors * (horizon - 1) / self.critic_batch_size))
        batch_idx = torch.arange(num_actors, device=self.device_train).chunk(num_batches)

        all_values = []
        with DataLoader(data, batch_idx, self.device_train, 4, dim=1) as data_loader:
            for batch_index in range(num_batches):
                batch = AttrDict(data_loader.get_next_batch())
                ac_out = self._train_model(
                    batch.states, actions=batch.actions, evaluate_heads=['state_values_1'])
                all_values.append(ac_out.state_values_1)

        data.values_old = torch.cat(all_values, 1).squeeze(-1).to(data.rewards.device)

    def _train_critic_on_batch(self, data):
        values_old = data.values_old
        value_targets = data.value_targets

        with torch.enable_grad():
            ac_out_cur = self._train_model(
                data.states, actions=data.actions, evaluate_heads=['state_values_1'])
            values_cur = ac_out_cur.state_values_1.squeeze(-1)

            v_pred_clipped = values_old + (values_cur - values_old).clamp(-self.value_clip, self.value_clip)
            vf_clip_loss = barron_loss(v_pred_clipped, value_targets, *self.barron_alpha_c, reduce=False)
            vf_nonclip_loss = barron_loss(values_cur, value_targets, *self.barron_alpha_c, reduce=False)

            loss = torch.max(vf_nonclip_loss, vf_clip_loss).mean()
            loss.backward()

        if self.grad_clip_norm is not None:
            clip_grad_norm_(self._train_model.parameters(), self.grad_clip_norm)
        self._critic_optimizer.step()
        self._train_model.zero_grad()

        return loss.detach()

    def _train_actor_on_batch(self, data):
        with torch.enable_grad():
            logits = self._train_model(data.states, evaluate_heads=['logits']).logits
            actions = self._train_model.heads.logits.pd.sample(logits)
            state_values = self._train_model(
                data.states, evaluate_heads=['state_values_1'], actions=actions).state_values_1
            loss = -state_values.mean()
            loss.backward()

        if self.grad_clip_norm is not None:
            clip_grad_norm_(self._train_model.parameters(), self.grad_clip_norm)
        self._actor_optimizer.step()
        self._train_model.zero_grad()

        return loss.detach()

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
                self.logger.add_image('state', img, self.frame)
            # vsize = data.value_targets.shape[-2] ** 0.5
            targets = data.value_targets
            values = data.values_old
            # v_mean = values.mean(-1)
            # t_mean = targets.mean(-1)
            self.logger.add_histogram('rewards', data.rewards, self.frame)
            self.logger.add_histogram('value_targets', targets, self.frame)
            # self.logger.add_histogram('advantages', data.advantages, self.frame)
            self.logger.add_histogram('values', values, self.frame)
            self.logger.add_scalar('value rmse', (values - targets).pow(2).mean().sqrt(), self.frame)
            self.logger.add_scalar('value abs err', (values - targets).abs().mean(), self.frame)
            self.logger.add_scalar('value max err', (values - targets).abs().max(), self.frame)
            if isinstance(self._train_model.heads.logits.pd, DiagGaussianPd):
                mean, std = data.logits.chunk(2, dim=1)
                self.logger.add_histogram('logits mean', mean, self.frame)
                self.logger.add_histogram('logits std', std, self.frame)
            elif isinstance(self._train_model.heads.logits.pd, CategoricalPd):
                self.logger.add_histogram('logits log_softmax', F.log_softmax(data.logits, dim=-1), self.frame)
            self.logger.add_histogram('logits', data.logits, self.frame)
            for name, param in self._train_model.named_parameters():
                self.logger.add_histogram(name, param, self.frame)