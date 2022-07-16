from abc import abstractmethod
from functools import partial
from typing import Tuple, Any, Dict, Iterable, Callable

import gym
import torch
from gym import register
from gym.envs.classic_control import CartPoleEnv
from ppo_pytorch.actors.actors import Actor
from ppo_pytorch.actors.fc_actors import create_ppo_fc_actor
from ppo_pytorch.algs.ppo import copy_state_dict
from torch import optim


class StatefulEnvMixin:
    def get_state(self):
        raise NotImplementedError

    def set_state(self, state):
        raise NotImplementedError


class CartPoleStatefulEnv(CartPoleEnv, StatefulEnvMixin):
    def get_state(self):
        return self.state, self.steps_beyond_done

    def set_state(self, state):
        self.state, self.steps_beyond_done = state


register(
    id='CartPoleStateful-v1',
    entry_point='ppo_pytorch.algs.alpha_zero:StatefulCartPoleEnv',
    max_episode_steps=500,
    reward_threshold=475.0,
)


class AlphaZero:
    def __init__(self, env_factory, num_envs,
                 reward_discount=0.99,
                 train_interval_frames=64 * 8,
                 batch_size=64,
                 model_factory: Callable = create_ppo_fc_actor,
                 optimizer_factory=partial(optim.Adam, lr=3e-4),
                 value_loss_scale=1.0,
                 pg_loss_scale=1.0,
                 entropy_loss_scale=0.01,
                 cuda_eval=True,
                 cuda_train=True,
                 reward_scale=1.0,
                 grad_clip_norm=None,
                 model_init_path=None):
        self._init_args = locals()
        self.reward_discount = reward_discount
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
        self.model_init_path = model_init_path

        self.envs = [env_factory() for _ in range(num_envs)]

        self._model: Actor = model_factory(self.envs[0].observation_space, self.envs[0].action_space)
        if self.model_init_path is not None:
            self._model.load_state_dict(torch.load(self.model_init_path))
            print(f'loaded model {self.model_init_path}')
        self._model = self._model.train().to(self.device_train)

        self._optimizer = optimizer_factory(self._model.parameters())

    def step(self):
        pass

    def train(self):
        pass