from collections import namedtuple
from itertools import count
from typing import List

import gym
from .multiplayer_env import MultiplayerEnv

EpisodeInfo = namedtuple('EpisodeInfo', 'reward, len')


class Monitor(gym.Wrapper):
    """
    Adds total reward and length stats to episode's last step `info`.
    """
    def __init__(self, env):
        super().__init__(env)
        self.episode_reward: float or List[float] = None
        self.episode_len: int or List[int] = None

    def reset(self, **kwargs):
        if isinstance(self.env, MultiplayerEnv):
            self.episode_reward = [0 for _ in range(self.env.num_players)]
            self.episode_len = [0 for _ in range(self.env.num_players)]
        else:
            self.episode_reward = 0
            self.episode_len = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        if isinstance(self.env, MultiplayerEnv):
            for i in range(self.env.num_players):
                self.episode_reward[i] += reward[i]
                self.episode_len[i] += 1
                if done:
                    self._add_episode_info(info[i], self.episode_reward[i], self.episode_len[i])
        else:
            self.episode_reward += reward
            self.episode_len += 1
            if done:
                self._add_episode_info(info, self.episode_reward, self.episode_len)

        return state, reward, done, info

    def _add_episode_info(self, info, reward, len):
        ep_orig = info.get('episode')
        if ep_orig is not None:
            info['episode_orig'] = ep_orig
        info['episode'] = EpisodeInfo(reward, len)

    def _warn_double_wrap(self):
        pass