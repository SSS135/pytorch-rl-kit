import os
import re
import tempfile
import time
from collections import deque, namedtuple
from typing import List, NamedTuple, Union, Dict, Tuple

from rl_exp.unity.variable_step_result import VariableStepResult
from torch.utils.tensorboard import SummaryWriter

from .monitor import EPISODE, EPISODE_ORIG

import numpy as np


class EpisodeInfo(NamedTuple):
    episode: int
    frame: int
    len: int
    true_reward: float


def get_valid_filename(s):
    """
    Return the given string converted to a string that can be used for a clean
    filename. Remove leading and trailing spaces; convert other spaces to
    underscores; and remove anything that is not an alphanumeric, dash,
    underscore, or dot.
    >>> get_valid_filename("john's portrait in 2004.jpg")
    'johns_portrait_in_2004.jpg'
    """
    s = str(s).strip().replace(' ', '_')
    return re.sub(r'(?u)[^-\w.]', '', s)


def get_log_dir(log_root_dir, alg_name, env_name, tag):
    timestr = time.strftime('%Y-%m-%d_%H-%M-%S')
    dir_name = f'{alg_name}_{get_valid_filename(env_name)}_{timestr}_{tag}_'
    return tempfile.mkdtemp('', dir_name, log_root_dir)


class TensorboardEnvLogger:
    def __init__(self,
                 alg_name,
                 env_name,
                 log_dir,
                 log_interval=10 * 1024,
                 reward_std_episodes=100,
                 tag='', ):
        """
        Tensorboard logger. Does logging of environment episode information
            and wrapping logger calls for classes inherited from `RLBase`.
        Args:
            alg_name: RL algorithm name
            env_name: Env name
            log_dir: Tensorboard logging path
            env_count: Number of parallely running envs.
            log_time_interval: Logging interval in seconds.
            reward_std_episodes: Reward statistics calculation window.
        """
        assert log_dir is not None
        self.log_interval = log_interval
        self.log_dir = log_dir
        self.alg_name = alg_name
        self.env_name = env_name
        self.tag = tag
        self.reward_window = deque(maxlen=reward_std_episodes)
        self.new_rewards: List[EpisodeInfo] = []
        self.new_rewards_orig: List[EpisodeInfo] = []
        self.episode = 0
        self.frame = 0
        self.last_log_frame = 0
        self.logger = SummaryWriter(log_dir)
        self.episodes_file = open(os.path.join(log_dir, 'episodes'), 'a')

    def step(self, data: Union[Dict, List[Dict], VariableStepResult], force_log: bool):
        if isinstance(data, VariableStepResult):
            info, frames = self._extract_info_variable(data)
        else:
            info, frames = self._extract_info_gym(data)
        self._step(info, frames, force_log)

    def _extract_info_gym(self, infos: Union[Dict, List[Dict]]) -> Tuple[List[EpisodeInfo], int]:
        if isinstance(infos, dict):
            infos = [infos]

        ep_infos = []
        for info in infos:
            ep_info = info.get(EPISODE)
            if ep_info is not None:
                ep_infos.append(EpisodeInfo(self.episode, self.frame, ep_info.len, ep_info.true_reward))

        return ep_infos, len(infos)

    def _extract_info_variable(self, data: VariableStepResult) -> Tuple[List[EpisodeInfo], int]:
        infos = []
        for done, len, reward in zip(data.done, data.episode_length, data.total_true_reward):
            if done:
                infos.append(EpisodeInfo(self.episode, self.frame, len, reward))
        return infos, data.agent_id.shape[0]

    def _step(self, infos: List[EpisodeInfo], num_step_frames: int, force_log: bool):
        self.frame += num_step_frames

        for ep_info in infos:
            ep_info = ep_info._replace(episode=self.episode, frame=self.frame)
            self.reward_window.append(ep_info.true_reward)
            self.new_rewards.append(ep_info)
            self.episode += 1
            self.episodes_file.write(f'{ep_info.true_reward}, {ep_info.len}\n')

        if len(self.new_rewards) != 0 and self.logger is not None and \
           (self.frame >= self.last_log_frame + self.log_interval or force_log):
            self.last_log_frame += self.log_interval
            wrmean = np.mean(self.reward_window)
            wrstd = np.std(self.reward_window)
            self.logger.add_scalar('Reward Window/Mean By Episode', wrmean, self.frame)
            self.logger.add_scalar('Reward Window/Std By Episode', wrstd, self.frame)
            self.logger.add_scalar('Reward Window/Norm Std By Episode', wrstd / max(1e-5, abs(wrmean)), self.frame)
            self.logger.add_scalar('Episode Lengths/Sample', self.new_rewards[-1].len, self.new_rewards[-1].episode)
            self.logger.add_scalar('Rewards By Episode/Sample', self.new_rewards[-1].true_reward, self.new_rewards[-1].episode)
            self.logger.add_scalar('Rewards By Frame/Sample', self.new_rewards[-1].true_reward, self.new_rewards[-1].frame)

            last_ep = self.new_rewards[-1].episode
            last_frame = self.new_rewards[-1].frame
            avg_len = np.mean([r.len for r in self.new_rewards])
            avg_r = np.mean([r.true_reward for r in self.new_rewards])
            self.logger.add_scalar('Episode Lengths/Average', avg_len, last_ep)
            self.logger.add_scalar('Rewards By Episode/Average By Episode', avg_r, last_ep)
            self.logger.add_scalar('Rewards By Frame/Average reward', avg_r, last_frame)
            self.logger.add_scalar('Rewards By Frame/Average len', avg_len, last_frame)

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
