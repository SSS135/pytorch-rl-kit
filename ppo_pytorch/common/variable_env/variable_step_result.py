import dataclasses
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class VariableStepResult:
    obs: np.ndarray
    rewards: np.ndarray
    reward_names: List[str]
    done: np.ndarray
    agent_id: np.ndarray
    true_reward: np.ndarray
    team_id: np.ndarray
    match_id: np.ndarray

    def copy(self, shallow=False):
        return VariableStepResult(**{k: (v if shallow else np.copy(v)) for k, v in dataclasses.asdict(self).items()})

    @staticmethod
    def concatenate(*res):
        return VariableStepResult(
            obs=np.concatenate([r.obs for r in res], 0),
            rewards=np.concatenate([r.rewards for r in res], 0),
            reward_names=res[0].reward_names,
            done=np.concatenate([r.done for r in res], 0),
            agent_id=np.concatenate([r.agent_id for r in res], 0),
            true_reward=np.concatenate([r.true_reward for r in res], 0),
            team_id=np.concatenate([r.team_id for r in res], 0),
            match_id=np.concatenate([r.match_id for r in res], 0),
        )