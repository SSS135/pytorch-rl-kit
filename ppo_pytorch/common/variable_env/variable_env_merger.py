from collections import defaultdict
from itertools import count
from typing import List, DefaultDict, Optional, Dict, Tuple, Set

import numpy as np

from .variable_step_result import VariableStepResult


class VariableEnvMerger:
    def __init__(self):
        self._actors_per_env: Optional[List[int]] = None
        self._env_to_global_agent_id: List[DefaultDict[int, int]] = []
        self._env_to_global_match_id: List[DefaultDict[int, int]] = []
        self._agent_team_match_id: Dict[int, Tuple[int, int]] = {}
        self._id_counter = count()
        self._match_agents: Dict[int, Set[int]] = defaultdict(lambda: set())

    def split_actions(self, actions: np.ndarray) -> List[np.ndarray]:
        return [ac for ac in np.split(np.asarray(actions), self._actors_per_env) if len(ac) != 0]

    def merge(self, data: List[VariableStepResult], env_ids: Optional[List[int]] = None) -> VariableStepResult:
        if env_ids is None:
            env_ids = list(range(len(data)))
        data = [x.copy(shallow=True) for x in data]
        self._create_id_mapping(env_ids)
        self._calc_actors_per_env(data)
        self._fix_agent_id(data, env_ids)
        res = VariableStepResult.concatenate(*data)
        self._validate_agent_id(res)
        return res

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

            for aid, mid, done in zip(raw_aid, raw_mid, env_res.done):
                self._match_agents[mid].add(aid)
                if done:
                    self._match_agents[mid].remove(aid)
                    if len(self._match_agents[mid]) == 0:
                        del self._match_agents[mid]
                        if mid in mid_map:
                            del mid_map[mid]

    def _validate_agent_id(self, data: VariableStepResult):
        for (a_id, t_id, m_id, done) in zip(data.agent_id.tolist(), data.team_id.tolist(), data.match_id.tolist(), data.done.tolist()):
            tm_id = self._agent_team_match_id.get(a_id)

            if tm_id is not None:
                assert tm_id[0] == t_id and tm_id[1] == m_id, f'done {done}, a_id {a_id}, t_id {t_id}, m_id {m_id}, tm_id {tm_id}'

            if done:
                if tm_id is not None:
                    del self._agent_team_match_id[a_id]
            else:
                if tm_id is None:
                    self._agent_team_match_id[a_id] = t_id, m_id