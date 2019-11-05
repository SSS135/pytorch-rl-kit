from collections import defaultdict, Iterable
from typing import List

import gym

from .multiplayer_env import MultiplayerEnv


class DefaultDictEx(defaultdict):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value


class Monitor(gym.Wrapper):
    """
    Adds total reward and length stats to episode's last step `info`.
    """
    def __init__(self, env):
        super().__init__(env)
        pnum = self.env.num_players if isinstance(self.env, MultiplayerEnv) else 1
        self._data = [DefaultDictEx(int) for _ in range(pnum)]

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        if isinstance(self.env, MultiplayerEnv):
            for i in range(self.env.num_players):
                self._add_step_info(self._data[i], info[i], reward[i])
                if done[i] if isinstance(done, Iterable) else done:
                    self._add_episode_info(info[i], self._data[i])
                    self._data[i] = DefaultDictEx(int)
        else:
            self._add_step_info(self._data[0], info, reward)
            if done:
                self._add_episode_info(info, self._data[0])
                self._data[0] = DefaultDictEx(int)

        return state, reward, done, info

    def _add_episode_info(self, info, data):
        ep_orig = info.get('episode')
        if ep_orig is not None:
            info['episode_orig'] = ep_orig
        info['episode'] = data

    def _add_step_info(self, data: dict, info: dict, reward: float):
        data['reward'] += reward
        data['len'] += 1
        reward_info = info.get('reward_info')
        if reward_info is not None:
            for k, v in reward_info.items():
                data[k] += v

    def _warn_double_wrap(self):
        pass