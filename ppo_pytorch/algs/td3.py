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
from torch.nn.utils import clip_grad_norm_
from torchvision.utils import make_grid

from .replay_buffer import ReplayBuffer
from .utils import lerp_module_
from ..actors import ModularActor, create_td3_fc_actor
from ..actors.utils import model_diff
from ..common.attr_dict import AttrDict
from ..common.barron_loss import barron_loss
from ..common.data_loader import DataLoader
from ..common.probability_distributions import DiagGaussianPd, CategoricalPd
from ..common.rl_base import RLBase


class TD3(RLBase):
    def __init__(self, observation_space, action_space,
                 reward_discount=0.99,
                 train_interval=16,
                 batch_size=128,
                 num_batches=16,
                 value_target_steps=1,
                 replay_buffer_size=128*1024,
                 target_model_blend=0.005,
                 train_noise_scale=0.2,
                 expl_noise_scale=0.1,
                 train_noise_clip=0.5,
                 critic_iters=2,
                 random_policy_frames=1000,
                 model_factory=create_td3_fc_actor,
                 actor_optimizer_factory=partial(optim.Adam, lr=3e-4),
                 critic_optimizer_factory=partial(optim.Adam, lr=3e-4),
                 cuda_eval=True,
                 cuda_train=True,
                 grad_clip_norm=None,
                 reward_scale=1.0,
                 # barron_alpha_c=(1.5, 1),
                 lr_scheduler_factory=None,
                 entropy_decay_factory=None,
                 # use_pop_art=False,
                 **kwargs):
        super().__init__(observation_space, action_space, **kwargs)
        self._init_args = locals()
        self.reward_discount = reward_discount
        self.train_interval = train_interval
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.value_target_steps = value_target_steps
        self.replay_buffer_size = replay_buffer_size
        self.target_model_blend = target_model_blend
        self.train_noise_scale = train_noise_scale
        self.expl_noise_scale = expl_noise_scale
        self.train_noise_clip = train_noise_clip
        self.critic_iters = critic_iters
        self.random_policy_frames = random_policy_frames
        self.model_factory = model_factory
        self.device_eval = torch.device('cuda' if cuda_eval else 'cpu')
        self.device_train = torch.device('cuda' if cuda_train else 'cpu')
        self.grad_clip_norm = grad_clip_norm
        self.reward_scale = reward_scale
        # self.barron_alpha_c = barron_alpha_c
        # self.use_pop_art = use_pop_art

        assert isinstance(action_space, gym.spaces.Box), action_space

        if self.model_init_path is None:
            self._train_model: ModularActor = model_factory(observation_space, action_space)
        else:
            self._train_model: ModularActor = torch.load(self.model_init_path)
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
        # self._pop_art = PopArt()
        # self._train_executor = ThreadPoolExecutor(max_workers=1)
        self._eval_steps = 0
        self._prev_data = None
        self._last_model_save_frame = 0
        self._train_future: Optional[Future] = None
        self._update_iter = 0

    def _step(self, rewards, dones, states) -> torch.Tensor:
        with torch.no_grad():
            # run network
            ac_out = self._eval_model(states.to(self.device_eval), evaluate_heads=['logits'])
            noise = self.expl_noise_scale * torch.randn(ac_out.logits.shape) if not self.disable_training else 0
            pd = self._eval_model.heads.logits.pd
            actions = (pd.sample(ac_out.logits).cpu() + noise).clamp(-pd.max_action, pd.max_action)
            if self.frame_eval < self.random_policy_frames:
                actions.data.uniform_(-pd.max_action, pd.max_action)

            if not self.disable_training:
                if self._prev_data is not None and rewards is not None:
                    self._replay_buffer.push(rewards=rewards, dones=dones, **self._prev_data)

                self._eval_steps += 1
                self._prev_data = dict(logits=ac_out.logits, states=states, actions=actions)

                min_replay_size = self.batch_size * (self.value_target_steps + 1)
                if self.frame_eval > 1024 and self._eval_steps >= self.train_interval and len(self._replay_buffer) >= min_replay_size:
                    self._eval_steps = 0
                    self._pre_train()
                    self._train()

            return actions

    def _pre_train(self):
        self.step_train = self.step_eval

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
            self._log_training_data(data)
            self._td3_update(data)
            self._model_saver.check_save_model(self._train_model, self.frame_train)

    def _create_data(self):
        # (steps, actors, *)
        data = self._replay_buffer.sample(self.batch_size * self.num_batches, self.value_target_steps + 1)
        data = AttrDict(data)
        data.rewards = self.reward_scale * data.rewards
        return data

    def _td3_update(self, data: AttrDict):
        num_rollouts = data.states.shape[1]

        data = AttrDict(states=data.states, actions=data.actions, rewards=data.rewards, dones=data.dones)

        rand_idx = torch.randperm(num_rollouts, device=self.device_train).chunk(self.num_batches)

        old_model = deepcopy(self._train_model)

        with DataLoader(data, rand_idx, self.device_train, 4, dim=1) as data_loader:
            for batch_index in range(self.num_batches):
                # prepare batch data
                batch = AttrDict(data_loader.get_next_batch())
                loss = self._batch_update(batch, self._do_log and batch_index == self.num_batches - 1)
                if self._do_actor_update:
                    lerp_module_(self._target_model, self._train_model, self.target_model_blend)
                self._update_iter += 1

        if self._do_log:
            self.logger.add_scalar('learning rate', self._actor_optimizer.param_groups[0]['lr'], self.frame_train)
            self.logger.add_scalar('total loss', loss, self.frame_train)
            self.logger.add_scalar('model abs diff', model_diff(old_model, self._train_model), self.frame_train)
            self.logger.add_scalar('model max diff', model_diff(old_model, self._train_model, True), self.frame_train)

        self._eval_model = deepcopy(self._train_model).to(self.device_eval).eval()

    @property
    def _do_actor_update(self):
        return self._update_iter % self.critic_iters == 0

    def _batch_update(self, batch, do_log=False):
        critc_loss = self._critic_step(batch, do_log)
        critc_loss.backward()
        if self.grad_clip_norm is not None:
            clip_grad_norm_(self._train_model.parameters(), self.grad_clip_norm)
        self._critic_optimizer.step()
        self._train_model.zero_grad()

        if self._do_actor_update:
            actor_loss = self._actor_step(batch, do_log)
            actor_loss.backward()
            if self.grad_clip_norm is not None:
                clip_grad_norm_(self._train_model.parameters(), self.grad_clip_norm)
            self._actor_optimizer.step()
            self._train_model.zero_grad()
        else:
            actor_loss = 0

        return actor_loss + critc_loss

    def _critic_step(self, data: AttrDict, do_log):
        actor_params = AttrDict()

        pd = self._target_model.heads.logits.pd
        logits = self._target_model(data.states[-1], evaluate_heads=['logits']).logits
        action_noise = torch.normal(0, self.train_noise_scale, size=logits.shape, device=self.device_train)\
            .clamp_(-self.train_noise_clip, self.train_noise_clip)
        actor_params.actions = (pd.sample(logits) + action_noise)\
            .clamp_(-pd.max_action, pd.max_action)

        ac_out = self._target_model(data.states[-1], evaluate_heads=['state_values_1', 'state_values_2'], **actor_params)
        targets = torch.min(ac_out.state_values_1, ac_out.state_values_2)

        rewards = data.rewards.unsqueeze(-1)
        dones = data.dones.unsqueeze(-1)
        for i in reversed(range(self.value_target_steps)):
            targets = rewards[i] + self.reward_discount * (1 - dones[i]) * targets

        actor_params = AttrDict()
        actor_params.actions = data.actions[0]

        with torch.enable_grad():
            ac_out = self._train_model(data.states[0], evaluate_heads=['state_values_1', 'state_values_2'], **actor_params)
            loss = F.mse_loss(ac_out.state_values_1, targets) + F.mse_loss(ac_out.state_values_2, targets)

        return loss

    def _actor_step(self, data: AttrDict, do_log):
        actor_params = AttrDict()

        with torch.enable_grad():
            logits = self._train_model(data.states[0], evaluate_heads=['logits']).logits
            actor_params.actions = self._train_model.heads.logits.pd.sample(logits)
            state_values = self._train_model(data.states[0], evaluate_heads=['state_values_1'], **actor_params).state_values_1
            return -state_values.mean()

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