import dataclasses
from collections import defaultdict
from itertools import count
from typing import List, DefaultDict, Optional

import numpy as np

from .variable_step_result import VariableStepResult


class VariableEnvMerger:
    def __init__(self):
        self._actors_per_env: Optional[List[int]] = None
        self._env_to_global_agent_id: List[DefaultDict[int, int]] = []
        self._env_to_global_match_id: List[DefaultDict[int, int]] = []
        self._id_counter = count()

    def split_actions(self, actions: np.ndarray) -> List[np.ndarray]:
        return np.split(np.asarray(actions), self._actors_per_env)

    def merge(self, data: List[VariableStepResult], env_ids: Optional[List[int]] = None) -> VariableStepResult:
        if env_ids is None:
            env_ids = list(range(len(data)))
        data = [x.copy(shallow=True) for x in data]
        self._create_id_mapping(env_ids)
        self._calc_actors_per_env(data)
        self._fix_agent_id(data, env_ids)
        return VariableStepResult.concatenate(*data)

    def _create_id_mapping(self, env_ids: List[int]):
        max_id = max(env_ids)
        while len(self._env_to_global_agent_id) <= max_id:
            self._env_to_global_agent_id.append(defaultdict(lambda: next(self._id_counter)))
            self._env_to_global_match_id.append(defaultdict(lambda: next(self._id_counter)))

    def _calc_actors_per_env(self, data: List[VariableStepResult]):
        self._actors_per_env = [len(data[0].agent_id)]
        for x in data[1:-1]:
            self._actors_per_env.append(len(x.agent_id) + self._actors_per_env[-1])

    def _fix_agent_id(self, data: List[VariableStepResult], env_ids: List[int]):
        for env_res, env_id in zip(data, env_ids):
            aid_map, mid_map = self._env_to_global_agent_id[env_id], self._env_to_global_match_id[env_id]
            raw_aid, raw_mid = env_res.agent_id, env_res.match_id
            env_res.agent_id = [aid_map[aid] for aid in env_res.agent_id]
            env_res.match_id = [mid_map[mid] for mid in env_res.match_id]

            for aid, done in zip(raw_aid, env_res.done):
                if done and aid in aid_map:
                    del aid_map[aid]

            mid_not_done = set()
            for mid, done in zip(raw_mid, env_res.done):
                if not done:
                    mid_not_done.add(mid)
            checked_mid = set()
            for mid in raw_mid:
                if mid not in checked_mid and mid not in mid_not_done:
                    del mid_map[mid]
                    checked_mid.add(mid)