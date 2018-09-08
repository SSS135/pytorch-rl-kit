import os
import time
from collections import deque, namedtuple
import tempfile

import numpy as np

Reward = namedtuple('Reward', 'reward, len, episode, frame')


class TensorboardEnvLogger:
    def __init__(self,
                 alg_name,
                 env_name,
                 log_path,
                 env_count,
                 log_interval=10 * 1024,
                 reward_std_episodes=100,
                 tag='',):
        """
        Tensorboard logger. Does logging of environment episode information
            and wrapping logger calls for classes inherited from `RLBase`.
        Args:
            alg_name: RL algorithm name
            env_name: Env name
            log_path: Tensorboard logging path
            env_count: Number of parallely running envs.
            log_time_interval: Logging interval in seconds.
            reward_std_episodes: Reward statistics calculation window.
        """
        assert log_path is not None
        from tensorboardX import SummaryWriter  # to remove dependency if not used
        self.log_interval = log_interval
        self.log_path = log_path
        self.alg_name = alg_name
        self.env_name = env_name
        self.tag = tag
        self.env_count = env_count
        self.reward_window = deque(maxlen=reward_std_episodes)
        self.reward_sum = np.zeros(self.env_count)
        self.episode_lens = np.zeros(self.env_count)
        self.new_rewards = []
        self.new_rewards_orig = []
        self.episode = 0
        self.frame = 0
        self.last_log_frame = 0
        timestr = time.strftime('%Y-%m-%d_%H-%M-%S')
        dir_name = f'{self.alg_name}_{self.env_name}_{tag}_{timestr}_'
        path = tempfile.mkdtemp('', dir_name, self.log_path)
        self.logger = SummaryWriter(path)
        self.episodes_file = open(os.path.join(path, 'episodes'), 'a')

    def step(self, infos: dict or list, force_log: bool):
        """
        Logging envs step results.
        Args:
            infos: List of infos returned by env.
            force_log: Do log regardless of `self.log_time_interval`. Usually True for last training frame.
        """
        if isinstance(infos, dict):
            infos = [infos]

        self.frame += self.env_count

        for info in infos:
            ep_info = info.get('episode')
            if ep_info is not None:
                self.reward_window.append(ep_info.reward)
                self.new_rewards.append(Reward(ep_info.reward, ep_info.len, self.episode, self.frame))
                self.episode += 1
                self.episodes_file.write(f'{ep_info.reward}, {ep_info.len}\n')
            ep_info_orig = info.get('episode_orig')
            if ep_info_orig is not None:
                self.new_rewards_orig.append(Reward(ep_info_orig.reward, ep_info_orig.len, self.episode, self.frame))

        if len(self.new_rewards) != 0 and self.logger is not None and \
           (self.frame >= self.last_log_frame + self.log_interval or force_log):
            self.last_log_frame += self.log_interval
            wrmean = np.mean(self.reward_window)
            wrstd = np.std(self.reward_window)
            self.logger.add_scalar('reward mean window by episode', wrmean, self.frame)
            self.logger.add_scalar('reward std window by episode', wrstd, self.frame)
            self.logger.add_scalar('reward norm std window by episode', wrstd / max(1e-5, abs(wrmean)), self.frame)
            self.logger.add_scalar('episode lengths', self.new_rewards[-1].len, self.new_rewards[-1].episode)
            self.logger.add_scalar('reward by episode', self.new_rewards[-1].reward, self.new_rewards[-1].episode)
            self.logger.add_scalar('reward by frame', self.new_rewards[-1].reward, self.new_rewards[-1].frame)
            avg_ep = np.mean([r.episode for r in self.new_rewards])
            avg_frame = np.mean([r.frame for r in self.new_rewards])
            avg_len = np.mean([r.len for r in self.new_rewards])
            avg_r = np.mean([r.reward for r in self.new_rewards])
            self.logger.add_scalar('avg episode lengths', avg_len, avg_ep)
            self.logger.add_scalar('avg reward by episode', avg_r, avg_ep)
            self.logger.add_scalar('avg reward by frame', avg_r, avg_frame)
            if len(self.new_rewards_orig) != 0:
                avg_r_orig = np.mean([r.reward for r in self.new_rewards_orig])
                self.logger.add_scalar('avg reward by frame orig', avg_r_orig, avg_frame)
                self.new_rewards_orig.clear()
            self.new_rewards.clear()
            self.episodes_file.flush()

    def add_scalar(self, *args, **kwargs):
        return self.logger.add_scalar(*args, **kwargs)

    def add_histogram(self, *args, **kwargs):
        return self.logger.add_histogram(*args, **kwargs)

    def add_image(self, *args, **kwargs):
        return self.logger.add_image(*args, **kwargs)

    def add_text(self, *args, **kwargs):
        return self.logger.add_text(*args, **kwargs)
