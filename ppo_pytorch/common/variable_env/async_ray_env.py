import time
from enum import Enum
import ray
from typing import List, NamedTuple, Any, Optional, Callable

import numpy as np

from .variable_env import VariableEnv
from .variable_env_merger import VariableEnvMerger
from .variable_step_result import VariableStepResult


TIMEOUT = 300


@ray.remote
class RemoteEnv:
    def __init__(self, env_fn, env_id):
        self.env_id = env_id
        self.env = env_fn()
        self.env.reset()

    def get_stats(self):
        return self.env.observation_space, self.env.action_space, self.env.env_name

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action), self.env_id

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


class AsyncRayEnv(VariableEnv):
    def __init__(self, env_factories: List[Callable], min_ready_envs=0.5):
        self.min_ready_envs = max(1, round(len(env_factories) * min_ready_envs))
        self._merger = VariableEnvMerger()
        self._awaiting_action_envs = []
        self._step_objs = []

        self._envs = [RemoteEnv.remote(ef, i) for i, ef in enumerate(env_factories)]
        self.observation_space, self.action_space, self.env_name = ray.get(self._envs[0].get_stats.remote())

    @property
    def num_envs(self):
        return len(self._envs)

    def step(self, action: np.ndarray) -> VariableStepResult:
        self._submit_actions(action)
        return self._get_states()

    def reset(self) -> VariableStepResult:
        self._awaiting_action_envs.clear()
        self._awaiting_action_envs.extend(self._envs)
        data = ray.get([env.reset.remote() for env in self._envs])
        return self._merger.merge(data, list(range(len(data))))

    def render(self):
        self._envs[0].render.remote()

    def close(self):
        ray.get([env.close.remote() for env in self._envs])

    def _submit_actions(self, actions: np.ndarray) -> None:
        actions = self._merger.split_actions(actions)
        assert len(self._awaiting_action_envs) == len(actions), (len(self._awaiting_action_envs), actions)
        for env, ac in zip(self._awaiting_action_envs, actions):
            self._step_objs.append(env.step.remote(ac))
        self._awaiting_action_envs.clear()

    def _get_states(self) -> VariableStepResult:
        assert len(self._awaiting_action_envs) == 0
        ready_objs, _ = ray.wait(self._step_objs, num_returns=self.min_ready_envs)
        states = ray.get(ready_objs)
        data, ids = [], []
        for obj, (state, i) in zip(ready_objs, states):
            env = self._envs[i]
            self._awaiting_action_envs.append(env)
            ids.append(i)
            data.append(state)
            self._step_objs.remove(obj)
        assert len(data) > 0
        return self._merger.merge(data, ids)