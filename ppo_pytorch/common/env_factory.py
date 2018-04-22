import gzip
import multiprocessing.dummy as mp
import os
import pprint
import retro
from itertools import count

import gym
import numpy as np
from retro.retro_env import RetroEnv

from .atari_wrappers import wrap_deepmind, make_atari
from .monitor import Monitor
from .rl_base import RLBase
from .tensorboard_env_logger import TensorboardEnvLogger
from .atari_wrappers import NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, FireResetEnv, ScaledFloatFrame, ClipRewardEnv, FrameStack
import gym.spaces as spaces
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from functools import partial
from .sonic_utils import SonicDiscretizer, AllowBacktracking, ChangeStateAtRestart
import random
from .sonic_utils import sonic_1_test_levels, sonic_2_test_levels, sonic_3_test_levels
from .sonic_utils import sonic_1_train_levels, sonic_2_train_levels, sonic_3_train_levels
from multiprocessing.dummy import Pool
import cv2


class NamedVecEnv:
    def __init__(self, env_name):
        self.env_name = env_name
        self.subproc_envs = None
        self.num_envs = None

        env = self.get_env_fn()()
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        env.close()

    def set_num_envs(self, num_envs):
        if self.subproc_envs is not None:
            self.subproc_envs.close()
        self.num_envs = num_envs
        self.subproc_envs = SubprocVecEnv([self.get_env_fn()] * num_envs)

    def step(self, actions):
        return self.subproc_envs.step(actions)

    def reset(self):
        return self.subproc_envs.reset()

    def get_env_fn(self):
        raise NotImplementedError


class FrameStackAtariEnvFactory(NamedVecEnv):
    def __init__(self, env_name, num_envs, episode_life=True, scale=True, clip_rewards=True, frame_stack=True):
        self.scale = scale
        self.clip_rewards = clip_rewards
        self.episode_life = episode_life
        self.frame_stack = frame_stack
        super().__init__(env_name)

    def get_env_fn(self):
        def make(env_name, episode_life, scale, clip_rewards, frame_stack):
            env = make_atari(env_name)
            env = Monitor(env)
            env = wrap_deepmind(env, episode_life, clip_rewards, frame_stack, scale)
            return env
        return partial(make, self.env_name, self.episode_life, self.scale, self.clip_rewards, self.frame_stack)


class SingleFrameAtariVecEnv(NamedVecEnv):
    def __init__(self, env_name, episode_life=True, scale=True, clip_rewards=True):
        self.scale = scale
        self.clip_rewards = clip_rewards
        self.episode_life = episode_life
        super().__init__(env_name)

    def get_env_fn(self):
        def make(env_name, episode_life, scale, clip_rewards):
            env = gym.make(env_name)
            assert 'NoFrameskip' in env.spec.id
            env = NoopResetEnv(env, noop_max=30)
            env = MaxAndSkipEnv(env, skip=4)
            env = Monitor(env)
            if episode_life:
                env = EpisodicLifeEnv(env)
            if 'FIRE' in env.unwrapped.get_action_meanings():
                env = FireResetEnv(env)
            env = ChannelTranspose(env)
            if scale:
                env = ScaledFloatFrame(env)
            if clip_rewards:
                env = ClipRewardEnv(env)
            return env
        return partial(make, self.env_name, self.episode_life, self.scale, self.clip_rewards)


class SonicVecEnv(NamedVecEnv):
    def __init__(self, game, state, scale=True, frame_stack=False, downscale=2, grayscale=True):
        self.scale = scale
        self.state = state
        self.frame_stack = frame_stack
        self.downscale = downscale
        self.grayscale = grayscale
        super().__init__(game)

    def get_env_fn(self):
        def make(game, state, scale, frame_stack, downscale, grayscale):
            from retro_contest.local import make
            env = make(game, state)
            env = Monitor(env)
            env = SonicDiscretizer(env)
            env = AllowBacktracking(env)
            env = Monitor(env)
            env = SimplifyFrame(env, downscale, grayscale)
            env = ChannelTranspose(env)
            if scale:
                env = ScaledFloatFrame(env)
            if frame_stack:
                env = FrameStack(env, 4)
            return env
        return partial(make, self.env_name, self.state, self.scale, self.frame_stack, self.downscale, self.grayscale)


class JointSonicVecEnv:
    sonic_names = ('SonicTheHedgehog-Genesis', 'SonicTheHedgehog2-Genesis', 'SonicAndKnuckles3-Genesis')

    def __init__(self, states='train', scale=True, frame_stack=False, downscale=2, grayscale=True):
        if states == 'train':
            states = [sonic_1_train_levels, sonic_2_train_levels, sonic_3_train_levels]
        assert len(states) == 3
        self.states = states
        self.scale = scale
        self.frame_stack = frame_stack
        self.downscale = downscale
        self.grayscale = grayscale
        self.env_name = 'Sonic123'
        self.pool = Pool(3)
        self.subproc_envs = None
        self.num_envs = None

        env = self.get_env_fn(self.sonic_names[0], states[0])()
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        env.close()

    def get_env_fn(self, game, states):
        def make(game, states, scale, frame_stack, downscale, grayscale):
            from retro_contest.local import make
            env = make(game, states[0])
            env = Monitor(env)
            env = SonicDiscretizer(env)
            env = AllowBacktracking(env)
            env = Monitor(env)
            env = ChangeStateAtRestart(env, states)
            env = SimplifyFrame(env, downscale, grayscale)
            env = ChannelTranspose(env)
            if scale:
                env = ScaledFloatFrame(env)
            if frame_stack:
                env = FrameStack(env, 4)
            return env
        return partial(make, game, states, self.scale, self.frame_stack, self.downscale, self.grayscale)

    def set_num_envs(self, num_envs):
        assert num_envs % 3 == 0
        if self.subproc_envs is not None:
            for e in self.subproc_envs:
                e.close()
        self.num_envs = num_envs
        self.subproc_envs = []
        for game, states in zip(self.sonic_names, self.states):
            self.env_name = game
            self.subproc_envs.append(SubprocVecEnv([self.get_env_fn(game, states)] * (num_envs // 3)))
        self.env_name = 'Sonic123'

    def step(self, actions):
        res = self.pool.starmap(lambda env, a: env.step(a), zip(self.subproc_envs, np.split(actions, 3, axis=0)))
        states, rewards, dones, infos = zip(*res)
        return np.concatenate(states), np.concatenate(rewards), np.concatenate(dones), [inf for arr in infos for inf in arr]

    def reset(self):
        return np.concatenate([env.reset() for env in self.subproc_envs], axis=0)


class SimpleEnvFactory(NamedVecEnv):
    def make_env(self):
        def make(env_name):
            env = gym.make(env_name)
            env = Monitor(env)
            return env
        return partial(make, self.env_name)


class ChannelTranspose(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(obs[2], obs[0], obs[1]))

    def _observation(self, frame):
        return frame.transpose(2, 0, 1)


class SimplifyFrame(gym.ObservationWrapper):
    def __init__(self, env, downscale=2, grayscale=True):
        super().__init__(env)
        self.downscale = downscale
        self.grayscale = grayscale
        ob = env.observation_space.shape
        assert len(ob) == 3 and ob[2] == 3
        assert ob[0] % downscale == 0 and ob[1] % downscale == 0
        out_ob = ob[0] // downscale, ob[1] // downscale, 1 if grayscale else 3
        self.observation_space = spaces.Box(low=0, high=255, shape=out_ob)

    def _observation(self, frame):
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if self.downscale != 1:
            resize_shape = self.observation_space.shape[1], self.observation_space.shape[0]
            frame = cv2.resize(frame, resize_shape, interpolation=cv2.INTER_AREA)
        return np.expand_dims(frame, -1) if self.grayscale else frame