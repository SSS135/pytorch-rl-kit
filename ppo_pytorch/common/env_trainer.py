import pprint
from itertools import count
from typing import Callable

import numpy as np
import torch

from .env_factory import NamedVecEnv
from .rl_base import RLBase
from .tensorboard_env_logger import TensorboardEnvLogger, get_log_dir


class EnvTrainer:
    def __init__(self,
                 rl_alg_factory: Callable[..., RLBase],
                 env_factory: Callable[[], NamedVecEnv],
                 log_root_path: str,
                 log_interval: int = 10000,
                 alg_name: str = 'RL',
                 tag: str = ''):
        """
        Simplifies training of RL algorithms with gym environments.
        Args:
            rl_alg_factory: RL algorithm factory / type.
            env: Environment factory.
                Accepted values are environment name, function which returns `gym.Env`, `gym.Env` object
            log_interval: Tensorboard logging interval in frames.
            log_root_path: Tensorboard output directory.
        """
        self._init_args = locals()
        self.rl_alg_factory = rl_alg_factory
        self.frame = 0
        self.all_rewards = []

        assert log_root_path is not None

        env = env_factory()
        env_name = env.env_name
        log_dir = get_log_dir(log_root_path, alg_name, env_name, tag)
        print('Log dir:', log_dir)
        env.set_num_envs(self.rl_alg_factory.keywords['num_actors'])
        self.states = env.reset()
        self.env = env

        self.rl_alg = rl_alg_factory(self.env.observation_space, self.env.action_space,
                                     log_interval=log_interval, model_save_folder=log_dir)

        self.logger = TensorboardEnvLogger(alg_name, env_name, log_dir, self.env.num_envs, log_interval, tag=tag)
        self.logger.add_text('EnvTrainer', pprint.pformat(self._init_args))
        self.rl_alg.logger = self.logger

    def step(self, always_log=False):
        """Do single step of RL alg"""

        self.states = torch.as_tensor(np.asarray(self.states, dtype=self.env.observation_space.dtype))

        # evaluate RL alg
        actions = self.rl_alg.eval(self.states)
        self.states, rewards, dones, infos = self.env.step(actions.numpy())

        rewards, dones = [torch.as_tensor(np.asarray(x, dtype=np.float32)) for x in (rewards, dones)]

        # process step results
        for info in infos:
            ep_info = info.get('episode')
            if ep_info is not None:
                self.all_rewards.append(ep_info)

        # send rewards and done flags to rl alg
        self.rl_alg.reward(rewards)
        self.rl_alg.finish_episodes(dones)

        self.frame += self.env.num_envs
        # logger step
        if self.logger is not None:
            self.logger.step(infos, always_log)

    def train(self, max_frames):
        """Train for specified number of frames and return episode info"""
        self.all_rewards = []
        while self.frame < max_frames:
            self.step()
        return self.all_rewards
