import multiprocessing.dummy as mp
import pprint
from itertools import count
from typing import Callable

import gym
import numpy as np

from .atari_wrappers import wrap_deepmind, make_atari
from .monitor import Monitor
from .rl_base import RLBase
from .tensorboard_env_logger import TensorboardEnvLogger
from .env_factory import EnvFactory


class GymWrapper:
    def __init__(self,
                 rl_alg_factory: Callable,
                 env_factory: EnvFactory,
                 log_time_interval=5,
                 log_path=None,):
        """
        Simplifies training of RL algorithms with gym environments.
        Args:
            rl_alg_factory: RL algorithm factory / type.
            env_factory: Environment factory.
                Accepted values are environment name, function which returns `gym.Env`, `gym.Env` object
            log_time_interval: Tensorboard logging interval in seconds.
            log_path: Tensorboard output directory.
        """
        self._init_args = locals()
        self.rl_alg_factory = rl_alg_factory
        self.env_factory = env_factory
        self.frame = 0

        self.rl_alg = rl_alg_factory(env_factory.observation_space, env_factory.action_space, log_time_interval=log_time_interval)
        self.envs = [env_factory.make_env() for _ in range(self.rl_alg.num_actors)]
        self.states = [env.reset() for env in self.envs]
        self.pool = mp.Pool(len(self.envs))
        self.all_rewards = []

        if log_path is not None:
            env_name = env_factory.env_name
            alg_name = type(self.rl_alg).__name__
            self.logger = TensorboardEnvLogger(alg_name, env_name, log_path, len(self.envs), log_time_interval)
            self.logger.add_text('GymWrapper', pprint.pformat(self._init_args))
            self.rl_alg.logger = self.logger
        else:
            self.logger = None

    def step(self, always_log=False):
        """Do single step of RL alg"""

        # evaluate RL alg
        actions = self.rl_alg.eval(self.states)
        self.states = []
        rewards = []
        dones = []
        infos = []

        # make env step
        def loop_fn(env, action):
            state, reward, done, info = env.step(action)
            if done:
                state = env.reset()
            return state, reward, done, info
        map = self.pool.starmap(loop_fn, zip(self.envs, actions))

        # process step results
        for i, (state, reward, done, info) in enumerate(map):
            self.states.append(state)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
            ep_info = info.get('episode')
            if ep_info is not None:
                self.all_rewards.append(ep_info)
        dones, rewards = np.asarray(dones), np.asarray(rewards)

        # send rewards and done flags to rl alg
        self.rl_alg.reward(rewards)
        self.rl_alg.finish_episodes(dones)

        self.frame += len(self.envs)
        # logger step
        if self.logger is not None:
            self.logger.step(infos, always_log)

    def train(self, max_frames):
        """Train for specified number of frames and return episode info"""
        self.all_rewards = []
        for _ in count():
            self.step(self.frame + len(self.envs) >= max_frames)
            if self.frame >= max_frames:
                break
        return self.all_rewards
