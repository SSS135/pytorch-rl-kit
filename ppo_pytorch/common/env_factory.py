import multiprocessing.dummy as mp
import pprint
from itertools import count

import gym
import numpy as np

from .atari_wrappers import wrap_deepmind, make_atari
from .monitor import Monitor
from .rl_base import RLBase
from .tensorboard_env_logger import TensorboardEnvLogger


class EnvFactory:
    def __init__(self, env_name):
        self.env_name = env_name
        env = self.make_env()
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        env.close()

    def make_env(self):
        raise NotImplementedError


class FrameStackAtariEnvFactory(EnvFactory):
    def __init__(self, env_name, episode_life=True, scale=True, clip_rewards=True, frame_stack=True):
        self.scale = scale
        self.clip_rewards = clip_rewards
        self.frame_stack = frame_stack
        self.episode_life = episode_life
        super().__init__(env_name)

    def make_env(self):
        env = make_atari(self.env_name)
        env = Monitor(env)
        env = wrap_deepmind(env, self.episode_life, self.clip_rewards, self.frame_stack, self.scale)
        return env


# class SingleFrameAtariEnvFactory(EnvFactory):
#     def __init__(self, env_name, episode_life=True, scale=True, clip_rewards=True, frame_stack=True):
#         self.scale = scale
#         self.clip_rewards = clip_rewards
#         self.frame_stack = frame_stack
#         self.episode_life = episode_life
#         super().__init__(env_name)
#
#     def make_env(self):
#         env = make_atari(self.env_name)
#         env = Monitor(env)
#         env = wrap_deepmind(env, self.episode_life, self.clip_rewards, self.frame_stack, self.scale)
#         return env


class SimpleEnvFactory(EnvFactory):
    def make_env(self):
        env = gym.make(self.env_name)
        env = Monitor(env)
        return env