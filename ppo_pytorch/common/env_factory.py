from functools import partial

import cv2
import gym
import gym.spaces as spaces
import numpy as np
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

from .threading_vec_env import ThreadingVecEnv
from .atari_wrappers import NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, FireResetEnv, ScaledFloatFrame, ClipRewardEnv, \
    FrameStack
from .monitor import Monitor


class NamedVecEnv:
    vec_env_types = dict(dummy=DummyVecEnv, thread=ThreadingVecEnv, process=SubprocVecEnv)

    def __init__(self, env_name, parallel='thread'):
        self.env_name = env_name
        self.parallel = parallel
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
        self.subproc_envs = self.vec_env_types[self.parallel]([self.get_env_fn()] * num_envs)

    def step(self, actions):
        return self.subproc_envs.step(actions)

    def reset(self):
        return self.subproc_envs.reset()

    def get_env_fn(self):
        raise NotImplementedError


class AtariVecEnv(NamedVecEnv):
    def __init__(self, env_name, episode_life=True, scale=True, clip_rewards=True,
                 frame_stack=True, grayscale=True, parallel=True):
        self.scale = scale
        self.clip_rewards = clip_rewards
        self.episode_life = episode_life
        self.frame_stack = frame_stack
        self.grayscale = grayscale
        super().__init__(env_name, parallel)

    def get_env_fn(self):
        def make(env_name, episode_life, scale, clip_rewards, frame_stack, grayscale):
            env = gym.make(env_name)
            assert 'NoFrameskip' in env.spec.id
            env = NoopResetEnv(env, noop_max=30)
            env = MaxAndSkipEnv(env, skip=4)
            env = Monitor(env)
            if episode_life:
                env = EpisodicLifeEnv(env)
            if 'FIRE' in env.unwrapped.get_action_meanings():
                env = FireResetEnv(env)
            env = SimplifyFrame(env, 84, grayscale)
            env = ChannelTranspose(env)
            if scale:
                env = ScaledFloatFrame(env)
            if clip_rewards:
                env = ClipRewardEnv(env)
            if frame_stack:
                env = FrameStack(env, 4)
            return env
        return partial(make, self.env_name, self.episode_life, self.scale, self.clip_rewards, self.frame_stack, self.grayscale)


class SimpleVecEnv(NamedVecEnv):
    def get_env_fn(self):
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

    def observation(self, frame):
        return frame.transpose(2, 0, 1)


class SimplifyFrame(gym.ObservationWrapper):
    def __init__(self, env, size=84, grayscale=True):
        super().__init__(env)
        self.size = size
        self.grayscale = grayscale
        ob = env.observation_space.shape
        assert len(ob) == 3 and ob[2] == 3
        out_ob = size, size, (1 if grayscale else 3)
        self.observation_space = spaces.Box(low=0, high=255, shape=out_ob)

    def observation(self, frame):
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # if self.downscale != 1:
        resize_shape = self.observation_space.shape[1], self.observation_space.shape[0]
        frame = cv2.resize(frame, resize_shape, interpolation=cv2.INTER_AREA)
        return np.expand_dims(frame, -1) if self.grayscale else frame