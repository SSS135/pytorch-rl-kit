from typing import List

import numpy as np

from .variable_env import VariableEnv
from .variable_env_merger import VariableEnvMerger
from .variable_step_result import VariableStepResult


class MergedVariableEnv(VariableEnv):
    def __init__(self, envs: List[VariableEnv]):
        self.envs = envs
        self.observation_space = envs[0].observation_space
        self.action_space = envs[0].action_space
        self.env_name = envs[0].env_name
        self._merger = VariableEnvMerger()

    def step(self, actions: np.ndarray) -> VariableStepResult:
        actions = self._merger.split_actions(actions)
        data = [env.step(a) for env, a in zip(self.envs, actions)]
        return self._merger.merge(data)

    def reset(self) -> VariableStepResult:
        data = [env.reset() for env in self.envs]
        return self._merger.merge(data)

    def render(self):
        self.envs[0].render()