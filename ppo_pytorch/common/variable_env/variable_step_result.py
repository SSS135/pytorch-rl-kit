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

    @staticmethod
    def concatenate(*res):
        return VariableStepResult(
            obs=np.concatenate([r.obs for r in res], 0),
            rewards=np.concatenate([r.rewards for r in res], 0),
            reward_names=res[0].reward_names,
            done=np.concatenate([r.done for r in res], 0),
            agent_id=np.concatenate([r.agent_id for r in res], 0),
            true_reward=np.concatenate([r.true_reward for r in res], 0),
        )