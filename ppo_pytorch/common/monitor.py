from collections import defaultdict
from typing import List

import gymnasium as gym


class DefaultAttrDict(defaultdict):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value


TRUE_REWARD = 'true_reward'
EP_LEN = 'len'
EPISODE = 'episode'
EPISODE_ORIG = 'episode_orig'


class Monitor(gym.Wrapper):
    """
    Adds total reward and length stats to episode's last step `info`.
    """
    def __init__(self, env):
        super().__init__(env)
        self._data = DefaultAttrDict(int)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        state, reward, term, trunc, info = self.env.step(action)
        done = term or trunc

        self._add_step_info(self._data, info, reward)
        if done:
            self._add_episode_info(info, self._data)
            self._data = DefaultAttrDict(int)

        return state, reward, done, info

    def _add_episode_info(self, info, data):
        ep_orig = info.get(EPISODE)
        if ep_orig is not None:
            info[EPISODE_ORIG] = ep_orig
        info[EPISODE] = data

    def _add_step_info(self, data: dict, info: dict, reward: float):
        data[TRUE_REWARD] += info[TRUE_REWARD] if TRUE_REWARD in info else reward
        data[EP_LEN] += 1