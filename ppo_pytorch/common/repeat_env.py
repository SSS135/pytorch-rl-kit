import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register

register(
    id='Repeat-v0',
    entry_point='ppo_pytorch.common.repeat_env:RepeatEnv',
)

register(
    id='RepeatNondeterministic-v0',
    entry_point='ppo_pytorch.common.repeat_env:RepeatNondeterministic',
)


class RepeatEnv(gym.Env):
    def __init__(self, deterministic=True):
        high = np.array([1])
        self.observation_space = spaces.Box(-high, high)
        self.action_space = spaces.Discrete(2)
        self.positive = False
        self.iter = None
        self.deterministic = deterministic

    def _step(self, action):
        assert self.iter is not None
        state = np.array([(1 if self.positive else -1) if self.deterministic else 0])
        if self.iter == 0:
            self.iter += 1
            return state, 0, False, {}
        elif self.iter == 1:
            r = 1 if self.positive == (action == 1) else -1
            self.iter = None
            return state, r, True, {}

    def _reset(self):
        self.iter = 0
        self.positive = random.random() > 0.5
        return np.array([1 if self.positive else -1])


class RepeatNondeterministic(RepeatEnv):
    def __init__(self):
        super().__init__(False)
