import pprint
from typing import Callable

import numpy as np
import torch

from .env_factory import NamedVecEnv
from .rl_base import RLBase
from .tensorboard_env_logger import TensorboardEnvLogger


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
        self._frame = 0
        self._rewards = self._done = None
        self._all_rewards = []

        assert log_root_path is not None

        env = env_factory()
        env_name = env.env_name
        log_dir = TensorboardEnvLogger.get_log_dir(log_root_path, alg_name, env_name, tag)
        print('Log dir:', log_dir)
        env.set_num_actors(self.rl_alg_factory.keywords['num_actors'])
        self._obs = env.reset()
        self._env = env

        self._rl_alg = rl_alg_factory(self._env.observation_space, self._env.action_space,
                                      log_interval=log_interval, model_save_folder=log_dir)

        self.logger = TensorboardEnvLogger(alg_name, env_name, log_dir, log_interval, tag=tag)
        self.logger.add_text('EnvTrainer', pprint.pformat(self._init_args))
        self._rl_alg.logger = self.logger

    def step(self, always_log=False):
        """Do single step of RL alg"""

        self._obs = torch.as_tensor(np.asarray(self._obs, dtype=self._env.observation_space.dtype))
        if self._rewards is None and self._done is None:
            self._rewards = self._done = torch.zeros(self._obs.shape[0])

        action = self._rl_alg.step(self._obs, self._rewards, self._done, None, None)
        self._obs, self._rewards, self._done, infos = self._env.step(action.numpy())

        self._rewards, self._done = [torch.as_tensor(x, dtype=torch.float32)
                                     for x in (self._rewards, self._done)]

        # process step results
        for info in infos:
            ep_info = info.get('episode')
            if ep_info is not None:
                self._all_rewards.append(ep_info)

        self._frame += self._env.num_actors
        # logger step
        if self.logger is not None:
            self.logger.step(infos, always_log)

    def train(self, max_frames):
        """Train for specified number of frames and return episode info"""
        self._all_rewards = []
        while self._frame < max_frames:
            self.step()
        return self._all_rewards

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # self._env.close()
        pass


