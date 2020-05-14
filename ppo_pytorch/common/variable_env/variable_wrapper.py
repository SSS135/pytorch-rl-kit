
import numpy as np

from .variable_env import VariableEnv
from .variable_step_result import VariableStepResult


class VariableWrapper(VariableEnv):
    def __init__(self, env: VariableEnv):
        self.env = env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.env_name = self.env.env_name

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.env, name)

    @classmethod
    def class_name(cls):
        return cls.__name__

    def step(self, action: np.ndarray) -> VariableStepResult:
        return self.env.step(action)

    def reset(self) -> VariableStepResult:
        return self.env.reset()

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)

    def __str__(self):
        return '<{}{}>'.format(type(self).__name__, self.env)

    def __repr__(self):
        return str(self)

    @property
    def unwrapped(self):
        return self.env.unwrapped


