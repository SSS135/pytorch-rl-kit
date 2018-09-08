from typing import List, Any

import numpy as np
import gym


class MultiplayerEnv(gym.Env):
    def __init__(self, num_players):
        self.num_players = num_players

    def reset(self) -> np.ndarray:
        raise NotImplementedError

    def step(self, action: List[Any]) -> (np.ndarray, np.ndarray, bool, dict):
        raise NotImplementedError

    def render(self, mode='human'):
        raise NotImplementedError
