from collections import namedtuple

import gym

EpisodeInfo = namedtuple('EpisodeInfo', 'reward, len')


class Monitor(gym.Wrapper):
    """
    Adds total reward and length stats to episode's last step `info`.
    """
    def __init__(self, env):
        super().__init__(env)
        self.episode_reward = 0
        self.episode_len = 0

    def _reset(self, **kwargs):
        self.episode_reward = 0
        self.episode_len = 0
        return self.env.reset(**kwargs)

    def _step(self, action):
        state, reward, done, info = self.env.step(action)

        self.episode_reward += reward
        self.episode_len += 1
        if done:
            info['episode'] = EpisodeInfo(self.episode_reward, self.episode_len)

        return state, reward, done, info