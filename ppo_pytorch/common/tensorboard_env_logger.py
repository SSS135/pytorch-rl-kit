import os
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Union, Dict, Tuple, Optional

from .variable_env.variable_step_result import VariableStepResult
from torch.utils.tensorboard import SummaryWriter
from .utils import get_valid_filename

from .monitor import EPISODE

import numpy as np


@dataclass
class EpisodeInfo:
    episode_index: int
    step_index: int
    len: int
    true_reward: float
    rewards: Optional[np.ndarray]


@dataclass
class CounterInfo:
    step: int
    true_reward: float
    rewards: Optional[np.ndarray]


class VariableEpisodeCounter:
    def __init__(self):
        self._episodes: Dict[int, CounterInfo] = {}

    def collect_episodes(self, data: VariableStepResult) -> List[CounterInfo]:
        episodes = []

        for done, agent_id, true_reward, rewards in zip(data.done, data.agent_id, data.true_reward, data.rewards):
            c_info = self._episodes.get(agent_id)
            if c_info is None:
                c_info = self._episodes[agent_id] = CounterInfo(-1, 0.0, np.zeros_like(rewards))

            c_info.step += 1
            c_info.true_reward += true_reward
            c_info.rewards += rewards

            if done:
                del self._episodes[agent_id]
                episodes.append(c_info)

        return episodes


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
        self._reward_window = deque(maxlen=reward_std_episodes)
        self._new_rewards: List[EpisodeInfo] = []
        self._new_rewards_orig: List[EpisodeInfo] = []
        self._episode = 0
        self._frame = 0
        self._last_log_frame = 0
        self._logger = SummaryWriter(log_dir)
        self._episodes_file = open(os.path.join(log_dir, 'episodes'), 'a')
        self._var_ep_counter = VariableEpisodeCounter()
        self._aux_reward_names = None

    def step(self, data: Union[Dict, List[Dict], VariableStepResult], force_log: bool):
        if isinstance(data, VariableStepResult):
            info, frames = self._extract_info_variable(data)
        else:
            info, frames = self._extract_info_gym(data)
        self._step(info, frames, force_log)

    def _extract_info_gym(self, infos: Union[Dict, List[Dict]]) -> Tuple[List[CounterInfo], int]:
        if isinstance(infos, dict):
            infos = [infos]

        ep_infos = []
        for info in infos:
            ep_info = info.get(EPISODE)
            if ep_info is not None:
                ep_infos.append(CounterInfo(ep_info.len, ep_info.true_reward, None))

        return ep_infos, len(infos)

    def _extract_info_variable(self, data: VariableStepResult) -> Tuple[List[CounterInfo], int]:
        self._aux_reward_names = data.reward_names
        return self._var_ep_counter.collect_episodes(data), data.agent_id.shape[0]

    def _step(self, infos: List[CounterInfo], num_step_frames: int, force_log: bool):
        self._frame += num_step_frames

        for ep_info in infos:
            ep_info = EpisodeInfo(self._episode, self._frame, ep_info.step, ep_info.true_reward, ep_info.rewards)
            self._reward_window.append(ep_info.true_reward)
            self._new_rewards.append(ep_info)
            self._episode += 1
            self._episodes_file.write(f'{ep_info.true_reward}, {ep_info.len}\n')

        if len(self._new_rewards) != 0 and self._logger is not None and \
           (self._frame >= self._last_log_frame + self.log_interval or force_log):
            self._last_log_frame += self.log_interval
            wrmean = np.mean(self._reward_window)
            wrstd = np.std(self._reward_window)
            self._logger.add_scalar('Reward Window/Mean By Episode', wrmean, self._frame)
            self._logger.add_scalar('Reward Window/Std By Episode', wrstd, self._frame)
            self._logger.add_scalar('Reward Window/Norm Std By Episode', wrstd / max(1e-5, abs(wrmean)), self._frame)
            self._logger.add_scalar('Episode Lengths/Sample', self._new_rewards[-1].len,
                                    self._new_rewards[-1].episode_index)
            self._logger.add_scalar('Rewards By Episode/Sample', self._new_rewards[-1].true_reward,
                                    self._new_rewards[-1].episode_index)
            self._logger.add_scalar('Rewards By Frame/Sample', self._new_rewards[-1].true_reward,
                                    self._new_rewards[-1].step_index)
            self._logger.add_histogram('Rewards By Frame/Rewards', np.array([r.true_reward for r in self._new_rewards]),
                                       self._new_rewards[-1].step_index)

            last_ep = self._new_rewards[-1].episode_index
            last_frame = self._new_rewards[-1].step_index
            avg_len = np.mean([r.len for r in self._new_rewards])
            avg_r = np.mean([r.true_reward for r in self._new_rewards])
            self._logger.add_scalar('Episode Lengths/Average', avg_len, last_ep)
            self._logger.add_scalar('Rewards By Episode/Average By Episode', avg_r, last_ep)
            self._logger.add_scalar('Rewards By Frame/Average reward', avg_r, last_frame)
            self._logger.add_scalar('Rewards By Frame/Average len', avg_len, last_frame)

            if self._new_rewards[0].rewards is not None:
                # (R, B)
                rewards = np.stack([r.rewards for r in self._new_rewards], 0).T
                for rname, r in zip(self._aux_reward_names, rewards):
                    self._logger.add_scalar(f'Rewards Aux/Average {rname}', np.mean(r), last_frame)

            self._new_rewards.clear()
            self._episodes_file.flush()

    def add_scalar(self, *args, **kwargs):
        return self._logger.add_scalar(*args, **kwargs)

    def add_histogram(self, *args, **kwargs):
        return self._logger.add_histogram(*args, **kwargs)

    def add_image(self, *args, **kwargs):
        return self._logger.add_image(*args, **kwargs)

    def add_text(self, *args, **kwargs):
        return self._logger.add_text(*args, **kwargs)

    @staticmethod
    def get_log_dir(log_root_dir, alg_name, env_name, tag):
        timestr = time.strftime('%Y-%m-%d_%H-%M-%S')
        dir_name = f'{alg_name}_{get_valid_filename(env_name)}_{timestr}_{tag}_'
        return tempfile.mkdtemp('', dir_name, log_root_dir)
