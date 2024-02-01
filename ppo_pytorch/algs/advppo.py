import math
import pprint
from asyncio import Future
from collections import deque
from concurrent.futures.thread import ThreadPoolExecutor
from copy import deepcopy
from enum import Enum
from functools import partial
from typing import Optional, Iterator, Deque

import gym.spaces
import numpy as np
import torch
import torch.autograd
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Parameter
from torch_optimizer import Adahessian

from ppo_pytorch.common.activation_norm import activation_norm_loss
from rl_exp.noisy_linear import NoisyLinear
from torch.nn.utils import clip_grad_norm_
from torchvision.utils import make_grid

from .ppo import copy_state_dict, SchedulerManager, log_training_data
from .steps_processor import StepsProcessor
from .utils import lerp_module_
from ..actors.fc_actors import create_ppo_fc_actor, create_advppo_fc_actor
from ..actors.actors import Actor, ModularActor
from ..actors.utils import model_diff
from ..common.attr_dict import AttrDict
from ..common.barron_loss import barron_loss
from ..common.data_loader import DataLoader
from ..common.pop_art import PopArt
from ..common.probability_distributions import DiagGaussianPd, CategoricalPd, FixedStdGaussianPd
from ..common.rl_base import RLBase, RLStepData
import torch.nn as nn


class AdvPPO(RLBase):
    def __init__(self, observation_space, action_space,
                 reward_discount=0.99,
                 advantage_discount=0.95,
                 horizon=64,
                 ppo_iters=10,
                 batch_size=64,
                 model_factory=create_advppo_fc_actor,
                 optimizer_factory=partial(optim.Adam, lr=3e-4),
                 value_clip: Optional[float] = None,
                 cuda_eval=False,
                 cuda_train=False,
                 grad_clip_norm=2,
                 reward_scale=1.0,
                 lr_scheduler_factory=None,
                 clip_decay_factory=None,
                 use_pop_art=False,
                 squash_values=False,
                 random_policy_frames=10000,
                 tested_actions=8,
                 gan_queue_len=4,
                 **kwargs):
        """
        Single threaded implementation of Proximal Policy Optimization Algorithms
        https://arxiv.org/pdf/1707.06347.pdf

        Args:
            observation_space (gym.Space): Environment's observation space
            action_space (gym.Space): Environment's action space
            reward_discount (float): Value function discount factor
            advantage_discount (float): Global Advantage Estimation discount factor
            horizon (int): Training will happen each `horizon` * `num_actors` steps
            ppo_iters (int): Number training passes for each state
            batch_size (int): Training batch size
            model_factory (Callable[[gym.Space, gym.Space], nn.Model]):
                Callable object receiving (observation_space, action_space) and returning actor-critic model
            optimizer_factory (Callable[[List[nn.Parameter]], optim.Optimizer]):
                Callable object receiving `model.parameters()` and returning model optimizer
            value_loss_scale (float): Multiplier for state-value loss
            entropy_loss_scale (float): Entropy maximization loss bonus (typically 0 to 0.01)
            policy_clip (float): policy clip strength
            value_clip (float): State-value clip strength
            cuda_eval (bool): Use CUDA for environment steps
            cuda_train (bool): Use CUDA for training steps
            grad_clip_norm (float or None): Max norm for gradient clipping (typically 0.5 to 40)
            reward_scale (float): Scale factor for environment's rewards
            lr_scheduler_factory (Callable[DecayLR]): Learning rate scheduler factory.
            clip_decay_factory (Callable[ValueDecay]): Policy / value clip scheduler factory.
            entropy_decay_factory (Callable[ValueDecay]): `entropy_loss_scale` scheduler factory.
            model_save_folder (str): Directory where models will be saved.
            model_save_tag (str): Tag is added to name of saved model. Used to save different models in one folder.
            model_save_interval (int): Interval in frames between model saves.
                Set to None to disable model saving.
            model_init_path (str): Path to model file to init from.
            save_intermediate_models (bool): If True, model saved at each `model_save_interval` frame
                is saved alongside new model. Otherwise it is overwritten by new model.
            num_actors (int): Number of parallel environments
            log_time_interval (float): Tensorboard logging interval in seconds
        """
        super().__init__(observation_space, action_space, **kwargs)
        self._init_args = locals()
        self.reward_discount = reward_discount
        self.advantage_discount = advantage_discount
        self.value_clip = value_clip
        self.horizon = horizon
        self.ppo_iters = ppo_iters
        self.batch_size = batch_size
        self.device_eval = torch.device('cuda' if cuda_eval else 'cpu')
        self.device_train = torch.device('cuda' if cuda_train else 'cpu')
        self.grad_clip_norm = grad_clip_norm
        self.model_factory = model_factory
        self.optimizer_factory = optimizer_factory
        self.reward_scale = reward_scale
        self.use_pop_art = use_pop_art
        self.lr_scheduler_factory = lr_scheduler_factory
        self.squash_values = squash_values
        self.random_policy_frames = random_policy_frames
        self.tested_actions = tested_actions
        self.gan_queue_len = gan_queue_len

        self._train_model: ModularActor = model_factory(observation_space, action_space)
        self._eval_model: ModularActor = model_factory(observation_space, action_space)
        if self.model_init_path is not None:
            self._train_model.load_state_dict(torch.load(self.model_init_path), True)
            print(f'loaded model {self.model_init_path}')
        copy_state_dict(self._train_model, self._eval_model)
        self._train_model = self._train_model.train().to(self.device_train, non_blocking=True)
        self._eval_model = self._eval_model.eval().to(self.device_eval, non_blocking=True)

        assert isinstance(self._train_model.heads.gen.pd, FixedStdGaussianPd)

        self._optimizer = optimizer_factory(self._train_model.parameters())
        self._scheduler = SchedulerManager(self._optimizer, lr_scheduler_factory, clip_decay_factory, None)
        self._last_model_save_frame = 0
        self._pop_art = PopArt()
        self._first_pop_art_update = True
        self._steps_processor = self._create_steps_processor()
        self._data_queue: Deque[AttrDict] = deque(maxlen=gan_queue_len)

    def _step(self, data: RLStepData) -> torch.Tensor:
        with torch.no_grad():
            states = data.obs.to(self.device_eval).unsqueeze(0).repeat(self.tested_actions, *[1] * data.obs.ndim)
            gen_actions = self._train_model(
                torch.cat([states, torch.randn_like(states)], dim=-1),
                evaluate_heads=['gen']).gen
            gen_actions = self._eval_model.heads.gen.pd.postprocess_action(gen_actions)
            q = self._train_model(states, actions=gen_actions, evaluate_heads=['q']).q
            q, max_index = q.max(0, keepdim=True)
            ac_index = max_index.repeat(*[1] * (max_index.ndim - 1), gen_actions.shape[-1])
            actions = gen_actions.gather(0, ac_index).squeeze(0)

            assert not torch.isnan(actions.sum())
            if self.frame_eval < self.random_policy_frames:
                actions.uniform_(-1, 1)
                q = self._train_model(data.obs.to(self.device_eval), actions=actions, evaluate_heads=['q']).q
            actions = actions.cpu()

            if not self.disable_training:
                q = q.squeeze(0).squeeze(-1)
                self._steps_processor.append_values(states=data.obs, rewards=data.rewards.sum(-1), dones=data.done,
                                                    actions=actions, state_values=q)

                if len(self._steps_processor.data.states) > self.horizon:
                    self._train()

            return actions

    def _train(self):
        with torch.no_grad():
            self.frame_train = self.frame_eval
            self._check_log()

            old_sp = self._steps_processor
            self._steps_processor = self._create_steps_processor()
            self._train_async(old_sp)

    def _create_data(self):
        self._steps_processor.complete()
        data = self._steps_processor.data
        self._steps_processor = self._create_steps_processor()
        return data

    def _train_async(self, steps_processor):
        with torch.no_grad():
            steps_processor.complete()
            data = steps_processor.data

            # log_training_data(self._do_log, self.logger, self.frame_train, self._train_model, data)
            self._ppo_update(data)
            self._model_saver.check_save_model(self._train_model, self.frame_train)
            self._scheduler.step(self.frame_train)

    def _ppo_update(self, data: AttrDict):
        self._apply_pop_art(data)

        data = AttrDict(states=data.states, state_values_old=data.state_values,
                        actions=data.actions, advantages=data.advantages, state_value_targets=data.state_value_targets)
        self._data_queue.append(data)

        batches = max(1, math.ceil(self.num_actors * self.horizon / self.batch_size))

        rand_idx = [torch.randperm(len(data.state_values_old), device=self.device_train) for _ in range(self.ppo_iters)]
        rand_idx = torch.cat(rand_idx, 0).chunk(batches * self.ppo_iters)

        old_model = deepcopy(self._train_model.state_dict())

        with DataLoader(data, rand_idx, self.device_train, 4) as data_loader:
            for ppo_iter in range(self.ppo_iters):
                for loader_iter in range(batches):
                    batch = AttrDict(data_loader.get_next_batch())
                    do_log = self._do_log and ppo_iter == self.ppo_iters - 1 and loader_iter == 0
                    ppo_loss = self._ppo_step(batch, do_log)

        gan_data = AttrDict(**{k: torch.cat([d[k] for d in self._data_queue], 0) for k in self._data_queue[0].keys()})
        rand_idx = torch.randperm(len(gan_data.state_values_old), device=self.device_train)
        rand_idx = rand_idx.chunk(len(gan_data.state_values_old) // self.batch_size)
        with DataLoader(gan_data, rand_idx, self.device_train, 4) as data_loader:
            for loader_iter in range(batches):
                batch = AttrDict(data_loader.get_next_batch())
                do_log = self._do_log and loader_iter == 0
                self._gan_step(batch, do_log)

        self._prev_max_ppo_iter = ppo_iter

        if self._do_log:
            self.logger.add_scalar('Optimizer/Learning Rate', self._learning_rate, self.frame_train)
            self.logger.add_scalar('Optimizer/Clip Mult', self._clip_mult, self.frame_train)
            self.logger.add_scalar('Losses/PPO Loss', ppo_loss, self.frame_train)
            self.logger.add_scalar('Stability/PPO Iters', ppo_iter + 1, self.frame_train)
            self.logger.add_scalar('Model Diff/Abs', model_diff(old_model, self._train_model), self.frame_train)
            self.logger.add_scalar('Model Diff/Max', model_diff(old_model, self._train_model, True), self.frame_train)

        self._unapply_pop_art()
        # NoisyLinear.randomize_network(self._train_model)
        # NoisyLinear.copy_noise(self._train_model, self._eval_model)

        copy_state_dict(self._train_model, self._eval_model)

    def _apply_pop_art(self, data):
        if self.use_pop_art:
            self._pop_art.update_statistics(data.state_value_targets)
            pa_mean, pa_std = self._pop_art.statistics
            if self._first_pop_art_update:
                self._first_pop_art_update = False
            else:
                self._train_model.heads.state_values.normalize(pa_mean, pa_std)
            data.state_values = (data.state_values - pa_mean) / pa_std
            data.state_value_targets = (data.state_value_targets - pa_mean) / pa_std
            if self._do_log:
                self.logger.add_scalar('PopArt/Mean', pa_mean, self.frame_train)
                self.logger.add_scalar('PopArt/Std', pa_std, self.frame_train)

    def _unapply_pop_art(self):
        if self.use_pop_art:
            self._train_model.heads.state_values.unnormalize(*self._pop_art.statistics)

    def _ppo_step(self, batch, do_log):
        with torch.enable_grad():
            actor_params = AttrDict()
            if do_log:
                actor_params.logger = self.logger
                actor_params.cur_step = self.frame_train

            actor_out = self._train_model(batch.states, actions=batch.actions, evaluate_heads=['q'], **actor_params)

            batch.state_values = actor_out.q.squeeze(-1)

            for k, v in list(batch.items()):
                batch[k] = v if k == 'states' or k == 'actions' else v.cpu()

            loss = self._get_value_loss(batch, do_log)

        loss.backward(create_graph=isinstance(self._optimizer, Adahessian))
        if self.grad_clip_norm is not None:
            clip_grad_norm_(self._train_model.parameters(), self.grad_clip_norm)
        self._optimizer.step()
        self._optimizer.zero_grad()

        return loss

    def _gan_step(self, batch, do_log):
        with torch.enable_grad():
            actor_params = AttrDict()
            if do_log:
                actor_params.logger = self.logger
                actor_params.cur_step = self.frame_train

            for k, v in list(batch.items()):
                batch[k] = v if k == 'states' or k == 'actions' else v.cpu()

            loss = self._get_gan_loss(batch, actor_params, do_log)

        loss.backward(create_graph=isinstance(self._optimizer, Adahessian))
        if self.grad_clip_norm is not None:
            clip_grad_norm_(self._train_model.parameters(), self.grad_clip_norm)
        self._optimizer.step()
        self._optimizer.zero_grad()

        return loss

    def _get_value_loss(self, batch, do_log=False):
        values, values_old = batch.state_values, batch.state_values_old
        state_value_targets = batch.state_value_targets

        # value loss
        if self.value_clip is not None:
            value_clip = self.value_clip * self._clip_mult
            v_pred_clipped = values_old + (values - values_old).clamp_(-value_clip, value_clip)
            vf_clip_loss = (v_pred_clipped - state_value_targets).pow_(2).mul_(0.5)
            vf_nonclip_loss = (values - state_value_targets).pow_(2).mul_(0.5)
            loss_value = torch.max(vf_nonclip_loss, vf_clip_loss)
            assert loss_value.shape == values.shape
        else:
            loss_value = (values - state_value_targets).pow_(2).mul_(0.5)

        loss_value = loss_value.mean()

        assert not np.isnan(loss_value.item()) and not np.isinf(loss_value.item()), loss_value.item()

        if do_log:
            with torch.no_grad():
                self.logger.add_scalar('Losses/State Value', loss_value.mean(), self.frame_train)

        return loss_value

    def _get_gan_loss(self, batch, actor_params, do_log=False):
        gen_actions = self._train_model(
            torch.cat([batch.states, torch.randn_like(batch.states)], dim=-1),
            evaluate_heads=['gen'], **actor_params).gen
        all_actions = torch.cat([batch.actions, gen_actions.detach()], dim=0)
        all_actions = all_actions + 0.1 * torch.randn_like(all_actions)
        disc_out = self._train_model(
            torch.cat([batch.states, batch.states], dim=0),
            actions=all_actions,
            evaluate_heads=['disc'], **actor_params).disc
        disc_out[batch.states.shape[0]:] *= -1
        disc_out.clamp_max(1.0)
        disc_loss = -disc_out.mean()
        disc_lt1 = (disc_out < 1).float().mean().item()
        if disc_lt1 < 0.2:
            disc_loss = 0.0 * disc_loss

        gen_disc_out = self._train_model(batch.states,
                                         actions=gen_actions + 0.1 * torch.randn_like(gen_actions),
                                         evaluate_heads=['disc'], **actor_params)\
            .disc.squeeze(-1)
        gen_grad = torch.autograd.grad(gen_disc_out.sum(), gen_actions)[0].detach()
        assert gen_actions.shape == gen_grad.shape
        gen_loss = -(gen_grad * gen_actions * (gen_disc_out.detach() < 1).float()).mean()
        gen_lt1 = (gen_disc_out < 1).float().mean().item()
        if gen_lt1 < 0.5:
            gen_loss = 0.0 * gen_loss

        if do_log:
            with torch.no_grad():
                self.logger.add_scalar('GAN/Disc All', disc_out.mean(), self.frame_train)
                self.logger.add_scalar('GAN/Disc Gen', gen_disc_out.mean(), self.frame_train)
                self.logger.add_scalar('GAN/Disc All LT1', disc_lt1, self.frame_train)
                self.logger.add_scalar('GAN/Disc Gen LT1', gen_lt1, self.frame_train)
                self.logger.add_scalar('Losses/Gen', gen_loss, self.frame_train)
                self.logger.add_scalar('Losses/Disc', disc_loss, self.frame_train)
                self.logger.add_scalar('Losses/Disc', gen_loss + disc_loss, self.frame_train)

        return disc_loss + gen_loss

    @property
    def _learning_rate(self):
        return self._optimizer.param_groups[0]['lr']

    @property
    def _clip_mult(self):
        return self._scheduler.clip_decay

    def _log_set(self):
        self.logger.add_text(self.__class__.__name__, pprint.pformat(self._init_args))
        self.logger.add_text('Model', str(self._train_model))

    def drop_collected_steps(self):
        self._steps_processor = self._create_steps_processor()

    def _create_steps_processor(self) -> StepsProcessor:
        return StepsProcessor(self._train_model.heads.gen.pd, self.reward_discount, self.advantage_discount,
                              self.reward_scale, True, self.squash_values)

    def __getstate__(self):
        d = dict(self.__dict__)
        d['_logger'] = None
        return d

    def __setstate__(self, d):
        self.__dict__ = d
