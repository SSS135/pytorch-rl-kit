import os
import os.path
import pprint
import random
import re
from collections import deque
from copy import deepcopy
from itertools import count
from typing import Callable, List, Deque

import numpy as np

from .env_factory import NamedVecEnv
from .rl_base import RLBase
from .tensorboard_env_logger import TensorboardEnvLogger


class MultiplayerEnvSelfPlayTrainer:
    def __init__(self,
                 rl_alg_factory: Callable,
                 env_factory: Callable[[], NamedVecEnv],
                 selection_train_frames: int,
                 num_archive_models: int,
                 model_archive_interval: int,
                 log_path: str,
                 log_interval=10 * 1024,
                 tag=''):
        """
        Simplifies training of RL algorithms with gym environments.
        Args:
            rl_alg_factory: RL algorithm factory / type.
            env: Environment factory.
                Accepted values are environment name, function which returns `gym.Env`, `gym.Env` object
            log_interval: Tensorboard logging interval in frames.
            log_path: Tensorboard output directory.
        """
        self._init_args = locals()
        self.rl_alg_factory = rl_alg_factory
        self.env_factory = env_factory
        self.selection_train_frames = selection_train_frames
        self.num_archive_models = num_archive_models
        self.model_archive_interval = model_archive_interval
        self.log_path = log_path
        self.log_interval = log_interval
        self.tag = tag

        self.env: NamedVecEnv = env_factory()
        self.frame: int = 0
        self.selected_alg_index = 0
        self.selection_change_frame = 0
        self.last_archive_frame = None
        self.selected_algs: List[RLBase] = None
        self.archive_algs: Deque[RLBase] = deque(maxlen=self.num_archive_models)

        self.rl_alg: RLBase = rl_alg_factory(self.env.observation_space, self.env.action_space, log_interval=log_interval)
        env_name = self.env.env_name
        alg_name = type(self.rl_alg).__name__
        self.env.set_num_actors(self.rl_alg.num_actors)
        self.logger = TensorboardEnvLogger(alg_name, env_name, log_path, self.env.num_actors, log_interval, tag=tag)
        self.logger.add_text('MPEnvSelPl', pprint.pformat(self._init_args))
        self.rl_alg.logger = self.logger

        self.all_states: np.ndarray = self.env.reset().transpose(1, 0, 2)
        self.num_players = self.all_states.shape[0]
        assert self.num_players <= self.num_archive_models + 1

    def step(self, always_log=False):
        """Do single step of RL alg"""

        self._try_archive_model()
        self._try_update_selection()

        # evaluate RL alg
        actions = [alg.eval(st) for (alg, st) in zip(self.selected_algs, self.all_states)]
        actions = np.array(actions).T
        data = self.env.step(actions)
        self.all_states, all_rewards, dones, all_infos = [np.asarray(v) for v in data]
        self.all_states, all_rewards, all_infos = self.all_states.transpose(1, 0, 2), all_rewards.T, all_infos.T

        assert self.all_states.shape[0] == all_rewards.shape[0]

        # send all_rewards and done flags to rl alg
        for alg, rewards, infos in zip(self.selected_algs, all_rewards, all_infos):
            alg._reward(rewards)
            alg.finish_episodes(dones)
            if alg == self.rl_alg:
                self.logger.step(infos, always_log)

        self.frame += self.env.num_actors

    def train(self, max_frames):
        """Train for specified number of frames and return episode info"""
        for _ in count():
            stop = self.frame + self.env.num_actors >= max_frames
            self.step(stop)
            if stop:
                break

    def _try_archive_model(self):
        if self.last_archive_frame is not None \
                and self.last_archive_frame + self.model_archive_interval > self.frame:
            return
        self.last_archive_frame = self.frame

        alg = deepcopy(self.rl_alg)
        alg.drop_collected_steps()
        alg.disable_training = True

        self.archive_algs.append(alg)
        while len(self.archive_algs) < self.num_players - 1:
            self.archive_algs.append(alg)

    def _try_update_selection(self):
        if self.selected_algs is not None \
                and self.selection_change_frame + self.selection_train_frames > self.frame:
            return
        self.selection_change_frame = self.frame

        if self.selected_algs is not None:
            for alg in self.selected_algs:
                alg.drop_collected_steps()

        indexes = np.random.choice(np.arange(len(self.archive_algs)), self.num_players - 1, replace=False)
        self.selected_algs = list(np.take(self.archive_algs, indexes))
        self.selected_algs.append(self.rl_alg)
        random.shuffle(self.selected_algs)

