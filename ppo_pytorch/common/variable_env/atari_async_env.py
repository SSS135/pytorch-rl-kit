from functools import partial

import gymnasium as gym
import numpy as np
from ppo_pytorch.common.atari_wrappers import NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, FireResetEnv, \
    FrameStack
from ppo_pytorch.common.env_factory import ChannelTranspose, SimplifyFrame
from ppo_pytorch.common.variable_env.async_variable_env import AsyncVariableEnv
from ppo_pytorch.common.variable_env.gym_to_variable_env import GymToVariableEnv
from ppo_pytorch.common.variable_env.variable_frame_stack import VariableFrameStack
from ppo_pytorch.common.variable_env.variable_wrapper import VariableWrapper


class VariableRescaleRewardEnv(VariableWrapper):
    eps = 1e-3

    def step(self, action):
        res = self.env.step(action)
        res.rewards = np.sign(res.rewards) * (np.sqrt(np.abs(res.rewards) + 1) - 1) + self.eps * res.rewards
        return res


class VariableClipRewardEnv(VariableWrapper):
    def step(self, action):
        """Bin reward to {+1, 0, -1} by its sign."""
        res = self.env.step(action)
        res.rewards = np.sign(res.rewards)
        return res


def atari_env_factory(env_name: str, frame_stack, episode_life, grayscale):
    env = gym.make(env_name)
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if episode_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = SimplifyFrame(env, 84, grayscale)
    env = ChannelTranspose(env)
    if frame_stack > 1:
        env = FrameStack(env, frame_stack)
    return GymToVariableEnv(env, env_name)


def make_atari_async_env(env_name: str, num_envs: int, min_ready_envs=0.5,
                         frame_stack=4, episode_life=True, clip_rewards=True, grayscale=True):
    env = AsyncVariableEnv([partial(atari_env_factory, env_name, frame_stack, episode_life, grayscale)] * num_envs, min_ready_envs)
    if clip_rewards:
        env = VariableClipRewardEnv(env)
    return env