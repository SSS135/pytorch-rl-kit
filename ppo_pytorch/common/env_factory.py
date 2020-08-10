from functools import partial

import cv2
import gym
import gym.spaces as spaces
import numpy as np
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

from .atari_wrappers import NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, FireResetEnv, ScaledFloatFrame, ClipRewardEnv, \
    FrameStack
from .monitor import Monitor
from .online_normalizer import OnlineNormalizer
from .threading_vec_env import ThreadingVecEnv


class NamedVecEnv:
    vec_env_types = dict(dummy=DummyVecEnv, thread=ThreadingVecEnv, process=SubprocVecEnv)

    def __init__(self, env_name: str, parallel: str = 'dummy', envs_per_process=1):
        self.env_name = env_name
        self.vec_env_cls = partial(SubprocVecEnv, in_series=envs_per_process) if parallel == 'process' \
            else self.vec_env_types[parallel]
        self.envs = None
        self.num_actors = None

        env = self.get_env_factory()()
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        env.close()

    def set_num_actors(self, num_actors):
        if self.envs is not None:
            self.envs.close()
        self.num_actors = num_actors
        self.envs = self.vec_env_cls([self.get_env_factory()] * num_actors)

    def step(self, actions):
        return self.envs.step(actions)

    def reset(self):
        return self.envs.reset()

    def get_env_factory(self):
        raise NotImplementedError


class AtariVecEnv(NamedVecEnv):
    def __init__(self, env_name, episode_life=True, scale_float_obs=False, clip_rewards=False,
                 frame_stack=True, grayscale=True, **kwargs):
        self.scale_float_obs = scale_float_obs
        self.clip_rewards = clip_rewards
        self.episode_life = episode_life
        self.frame_stack = frame_stack
        self.grayscale = grayscale
        super().__init__(env_name, **kwargs)

    def get_env_factory(self):
        def make(env_name, episode_life, scale_float_obs, clip_rewards, frame_stack, grayscale):
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
            if scale_float_obs:
                env = ScaledFloatFrame(env)
            if clip_rewards:
                env = ClipRewardEnv(env)
            if frame_stack:
                env = FrameStack(env, 4)
            return env

        return partial(make, self.env_name, self.episode_life, self.scale_float_obs, self.clip_rewards,
                       self.frame_stack, self.grayscale)


class SimpleVecEnv(NamedVecEnv):
    def __init__(self, env_name, observation_norm=False, frame_skip=1, parallel='thread'):
        self.observation_norm = observation_norm
        self.frame_skip = frame_skip
        super().__init__(env_name, parallel)

    def get_env_factory(self):
        def make(env_name, observation_norm, frame_skip):
            env = gym.make(env_name)
            env = Monitor(env)
            if frame_skip > 1:
                env = FrameSkipEnv(env, frame_skip)
            if observation_norm:
                env = ObservationNorm(env)
            return env

        return partial(make, self.env_name, self.observation_norm, self.frame_skip)


class ProcgenVecEnv(NamedVecEnv):
    def __init__(self, env_name, frame_skip=1, parallel='thread'):
        self.frame_skip = frame_skip
        super().__init__(env_name, parallel)

    def get_env_factory(self):
        def make(env_name, frame_skip):
            env = gym.make(env_name)
            env = ChannelTranspose(env)
            env = Monitor(env)
            if frame_skip > 1:
                env = FrameSkipEnv(env, frame_skip)
            return env

        return partial(make, self.env_name, self.frame_skip)


class FrameSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        self.skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0
        done = None
        for i in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, np.asarray(total_reward).astype(np.float32), done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class ChannelTranspose(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs = env.observation_space.shape
        low, high, dtype = env.observation_space.low[0, 0, 0], env.observation_space.high[0, 0, 0], env.observation_space.dtype
        self.observation_space = spaces.Box(low=low, high=high, shape=(obs[2], obs[0], obs[1]), dtype=dtype)

    def observation(self, frame):
        return np.moveaxis(np.asarray(frame), -1, -3)


class FloatToByteObs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=obs, dtype=np.uint8)

    def observation(self, frame):
        frame = np.asarray(frame)
        out = np.empty(frame.shape, np.uint8)
        np.multiply(frame, 255, out=out, casting='unsafe')
        return out


class SimplifyFrame(gym.ObservationWrapper):
    def __init__(self, env, size=84, grayscale=True):
        super().__init__(env)
        self.size = size
        self.grayscale = grayscale
        ob = env.observation_space.shape
        assert len(ob) == 3 and ob[2] == 3
        out_ob = size, size, (1 if grayscale else 3)
        self.observation_space = spaces.Box(low=0, high=255, shape=out_ob, dtype=np.uint8)

    def observation(self, frame):
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # if self.downscale != 1:
        resize_shape = self.observation_space.shape[1], self.observation_space.shape[0]
        frame = cv2.resize(frame, resize_shape, interpolation=cv2.INTER_AREA)
        return np.expand_dims(frame, -1) if self.grayscale else frame


class ObservationNorm(gym.ObservationWrapper):
    def __init__(self, env, eps=(1e-3, 1e5), absmax=5, scale=True, center=True, single_value=False):
        super().__init__(env)
        self._norm = OnlineNormalizer(eps, absmax, scale, center, single_value)
        obs = env.observation_space.shape
        self.observation_space = spaces.Box(low=-absmax, high=absmax, shape=obs, dtype=np.float32)

    def observation(self, frame):
        return self._norm(np.asarray(frame))
