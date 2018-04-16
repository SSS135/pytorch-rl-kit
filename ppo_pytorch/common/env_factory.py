import multiprocessing.dummy as mp
import pprint
from itertools import count

import gym
import numpy as np

from .atari_wrappers import wrap_deepmind, make_atari
from .monitor import Monitor
from .rl_base import RLBase
from .tensorboard_env_logger import TensorboardEnvLogger
from .atari_wrappers import NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, FireResetEnv, ScaledFloatFrame, ClipRewardEnv
import gym.spaces as spaces
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from functools import partial
from .sonic_utils import SonicDiscretizer, AllowBacktracking


class NamedVecEnv:
    def __init__(self, env_name):
        self.env_name = env_name
        self.subproc_env = None
        self.num_envs = None

        env = self.get_env_fn()()
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        env.close()

    def set_num_envs(self, num_envs):
        if self.subproc_env is not None:
            self.subproc_env.close()
        self.num_envs = num_envs
        self.subproc_env = SubprocVecEnv([self.get_env_fn()] * num_envs)

    def step(self, actions):
        return self.subproc_env.step(actions)

    def reset(self):
        return self.subproc_env.reset()

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
    def __init__(self, game, state, scale=True):
        self.scale = scale
        self.state = state
        super().__init__(game)

    def get_env_fn(self):
        def make(game, state, scale):
            from retro_contest.local import make
            env = make(game, state)
            # env = NoopResetEnv(env, noop_max=30)
            # env = MaxAndSkipEnv(env, skip=4)
            env = Monitor(env)
            # if episode_life:
            #     env = EpisodicLifeEnv(env)
            # if 'FIRE' in env.unwrapped.get_action_meanings():
            #     env = FireResetEnv(env)
            env = SonicDiscretizer(env)
            env = AllowBacktracking(env)
            env = ChannelTranspose(env)
            if scale:
                env = ScaledFloatFrame(env)
            return env
        return partial(make, self.env_name, self.state, self.scale)


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