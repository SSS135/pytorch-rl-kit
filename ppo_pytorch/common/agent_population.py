from typing import List, Dict, Callable
from .rl_base import RLBase
from copy import deepcopy
import numpy as np


class PopAgent:
    def __init__(self, rl: RLBase, params: np.ndarray):
        self.rl = rl
        self.fitness = 0
        self.params = params


class AgentPopulation:
    def __init__(self, pops: List[RLBase], param_init: Dict[str, Callable[[], float]], param_reset_chance=0.1):
        self.param_init = param_init
        self.param_reset_chance = param_reset_chance
        self.pops = [PopAgent(rl, self._create_new_parameters()) for rl in pops]
        self._returned_pops = None

    def get_pops(self, count: int) -> List[PopAgent]:
        assert self._returned_pops is None

    def reward_pops(self, rewards: np.ndarray):
        assert self._returned_pops is not None

    def _create_new_parameters(self) -> np.ndarray:
        return np.array([v() for k, v in self.param_init])