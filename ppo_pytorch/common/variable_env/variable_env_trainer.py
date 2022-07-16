import pprint
from typing import Callable

import torch

from .variable_env import VariableVecEnv
from ..rl_base import RLBase
from ..tensorboard_env_logger import TensorboardEnvLogger


class VariableEnvTrainer:
    def __init__(self,
                 rl_alg_factory: Callable[..., RLBase],
                 env_factory: Callable[[], VariableVecEnv],
                 log_root_path: str,
                 log_interval: int = 10000,
                 alg_name: str = 'RL',
                 tag: str = ''):
        """
        Simplifies training of RL algorithms with gym environments.
        Args:
            rl_alg_factory: RL algorithm factory / type.
            env: Environment factory.
            log_interval: Tensorboard logging interval in frames.
            log_root_path: Tensorboard output directory.
        """
        self._init_args = locals()
        self.rl_alg_factory = rl_alg_factory
        self._frame = 0

        assert log_root_path is not None

        self._env = env_factory()
        log_dir = TensorboardEnvLogger.get_log_dir(log_root_path, alg_name, self._env.env_name, tag)
        print('Log dir:', log_dir)
        self._data = self._env.reset()

        self._rl_alg = rl_alg_factory(
            self._env.observation_space, self._env.action_space, num_rewards=len(self._data.reward_names),
            log_interval=log_interval, model_save_folder=log_dir)

        self.logger = TensorboardEnvLogger(alg_name, self._env.env_name, log_dir, log_interval, tag=tag)
        self.logger.add_text('EnvTrainer', pprint.pformat(self._init_args))
        self._rl_alg.logger = self.logger

    def step(self, always_log=False):
        """Do single step of RL alg"""

        tensors = (self._data.obs, self._data.rewards, self._data.done,
                   self._data.true_reward, self._data.agent_id, self._data.action_mask)
        obs, rewards, done, true_reward, agent_id, action_mask = [torch.as_tensor(x) for x in tensors]

        action = self._rl_alg.step(obs, rewards, done, true_reward, agent_id, action_mask)
        self._data = self._env.step(action.numpy())

        self._frame += len(self._data.obs)
        # logger step
        if self.logger is not None:
            self.logger.step(self._data, always_log)

    def train(self, max_frames):
        while self._frame < max_frames:
            self.step()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._env.close()