import dataclasses
from collections import defaultdict
from itertools import count
from typing import List, DefaultDict

import numpy as np

from .variable_step_result import VariableStepResult


class VariableEnvMerger:
    def __init__(self):
        self._actors_per_env = None
        self._env_to_global_agent_id: List[DefaultDict[int, int]] = []
        self._agent_id_counter = count()

    def split_actions(self, actions: np.ndarray) -> List[np.ndarray]:
        return np.split(np.asarray(actions), self._actors_per_env)

    def merge(self, data: List[VariableStepResult]) -> VariableStepResult:
        self._create_id_mapping(data)
        self._calc_actors_per_env(data)
        self._fix_agent_id(data)
        return self._cat_env_data(data)

    def _create_id_mapping(self, data: List[VariableStepResult]):
        assert len(self._env_to_global_agent_id) in (0, len(data)), (len(self._env_to_global_agent_id), len(data))
        if len(self._env_to_global_agent_id) == 0:
            for _ in range(len(data)):
                self._env_to_global_agent_id.append(defaultdict(lambda: next(self._agent_id_counter)))

    def _calc_actors_per_env(self, data: List[VariableStepResult]):
        self._actors_per_env = [len(data[0].agent_id)]
        for x in data[1:-1]:
            self._actors_per_env.append(len(x.agent_id) + self._actors_per_env[-1])

    def _fix_agent_id(self, data: List[VariableStepResult]):
        for env_res, aid_map in zip(data, self._env_to_global_agent_id):
            raw_aid = env_res.agent_id
            env_res.agent_id = [aid_map[aid] for aid in env_res.agent_id]
            for aid, done in zip(raw_aid, env_res.done):
                if done and aid in aid_map:
                    del aid_map[aid]

    def _cat_env_data(self, data: List[VariableStepResult]) -> VariableStepResult:
        tuples = [dataclasses.astuple(x) for x in data]
        step = VariableStepResult(*[np.concatenate(x, 0) for x in zip(*tuples)])
        step.reward_names = data[0].reward_names
        assert len(step.agent_id) == len(set(step.agent_id)), step.agent_id
        assert len(step.agent_id) > 0
        return step