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

    def reset(self, **kwargs):
        self.episode_reward = 0
        self.episode_len = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        self.episode_reward += reward
        self.episode_len += 1
        if done:
            ep_orig = info.get('episode')
            if ep_orig is not None:
                info['episode_orig'] = ep_orig
            info['episode'] = EpisodeInfo(self.episode_reward, self.episode_len)

        return state, reward, done, info

    def _warn_double_wrap(self):
        pass