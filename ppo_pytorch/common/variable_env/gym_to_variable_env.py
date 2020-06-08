from functools import partial

import gym
import numpy as np
from ppo_pytorch.common.env_factory import FrameSkipEnv, ChannelTranspose
from ppo_pytorch.common.variable_env.async_variable_env import AsyncVariableEnv
from ppo_pytorch.common.variable_env.variable_env import VariableEnv
from ppo_pytorch.common.variable_env.variable_frame_stack import VariableFrameStack
from ppo_pytorch.common.variable_env.variable_step_result import VariableStepResult


def env_factory(env_name: str, frame_skip: int, channel_transpose: bool):
    env = gym.make(env_name)
    if channel_transpose:
        env = ChannelTranspose(env)
    if frame_skip > 1:
        env = FrameSkipEnv(env, frame_skip)
    return GymToVariableEnv(env, env_name)


def make_async_env(env_name: str, num_envs: int, frame_skip=1, frame_stack=1, min_ready_envs=0.5, channel_transpose=False):
    env = AsyncVariableEnv([partial(env_factory, env_name, frame_skip, channel_transpose)] * num_envs, min_ready_envs)
    if frame_stack > 1:
        env = VariableFrameStack(env, k=frame_stack)
    return env


class GymToVariableEnv(VariableEnv):
    def __init__(self, env: gym.Env, env_name: str):
        self.env = env
        self.env_name = env_name
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self._actor_id = 0

    def step(self, action: np.ndarray) -> VariableStepResult:
        state, reward, done, info = self.env.step(action[0])
        state, reward, done = [np.asarray([x]) for x in (state, reward, done)]
        res = VariableStepResult(
            obs=state,
            rewards=np.expand_dims(reward, axis=-1),
            reward_names=['True Reward'],
            done=done,
            agent_id=np.full(1, self._actor_id, dtype=np.int64),
            true_reward=reward,
            team_id=np.zeros(1, dtype=np.int64),
            match_id=np.full(1, self._actor_id, dtype=np.int64),
        )
        if done[0]:
            res = VariableStepResult.concatenate(res, self.reset())
        return res

    def reset(self) -> VariableStepResult:
        self._actor_id += 1
        state = self.env.reset()
        state = np.asarray([state])
        return VariableStepResult(
            obs=state,
            rewards=np.zeros((1, 1), dtype=np.float32),
            reward_names=['True Reward'],
            done=np.zeros(1, dtype=np.float32),
            agent_id=np.full(1, self._actor_id, dtype=np.int64),
            true_reward=np.zeros(1, dtype=np.float32),
            team_id=np.zeros(1, dtype=np.int64),
            match_id=np.full(1, self._actor_id, dtype=np.int64),
        )

    def close(self):
        self.env.close()

    def render(self):
        self.env.render()