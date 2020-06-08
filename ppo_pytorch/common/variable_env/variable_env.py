from abc import abstractmethod, ABCMeta
from typing import List

import gym
import numpy as np

from .variable_step_result import VariableStepResult


class VariableEnv(metaclass=ABCMeta):
    action_space: gym.Space = None
    observation_space: gym.Space = None
    env_name: str = None

    @abstractmethod
    def step(self, action: np.ndarray) -> VariableStepResult:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> VariableStepResult:
        raise NotImplementedError

    def render(self):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        return

    @property
    def unwrapped(self):
        return self

    def __str__(self):
        return '<{} instance>'.format(type(self).__name__)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        return False


class VariableVecEnv(VariableEnv):
    num_envs: int = None
