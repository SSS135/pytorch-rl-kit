import pprint
from itertools import count
from typing import Callable

import numpy as np

from .tensorboard_env_logger import TensorboardEnvLogger
import torch


class EnvTrainer:
    def __init__(self,
                 rl_alg_factory: Callable,
                 env_factory: Callable,
                 log_interval=10 * 1024,
                 log_path=None,
                 tag=''):
        """
        Simplifies training of RL algorithms with gym environments.
        Args:
            rl_alg_factory: RL algorithm factory / type.
            env: Environment factory.
                Accepted values are environment name, function which returns `gym.Env`, `gym.Env` object
            log_interval: Tensorboard logging interval in frames.
            log_path: Tensorboard output directory.
        """
        self._init_args = locals()
        self.rl_alg_factory = rl_alg_factory
        self.env = env_factory()
        self.frame = 0

        self.rl_alg = rl_alg_factory(self.env.observation_space, self.env.action_space, log_interval=log_interval)
        self.env.set_num_envs(self.rl_alg.num_actors)
        self.states = self.env.reset()
        self.all_rewards = []

        if log_path is not None:
            env_name = self.env.env_name
            alg_name = type(self.rl_alg).__name__
            self.logger = TensorboardEnvLogger(alg_name, env_name, log_path, self.env.num_envs, log_interval, tag=tag)
            self.logger.add_text('EnvTrainer', pprint.pformat(self._init_args))
            self.rl_alg.logger = self.logger
        else:
            self.logger = None

    def step(self, always_log=False):
        """Do single step of RL alg"""

        # evaluate RL alg
        actions = self.rl_alg.eval(self.states)
        self.states, rewards, dones, infos = self.env.step(actions.numpy())

        self.states, rewards, dones = [torch.as_tensor(np.asarray(x, dtype=np.float32))
                                       for x in (self.states, rewards, dones)]

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
        for _ in count():
            self.step(self.frame + self.env.num_envs >= max_frames)
            if self.frame >= max_frames:
                break
        # return self.all_rewards
