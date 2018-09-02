import time
from enum import Enum

import gym
import numpy as np
from gym.spaces import Discrete


class RLStep(Enum):
    """Internal state of `RLBase`"""
    EVAL = 0
    REWARD = 1
    FINISH = 2


class RLBase:
    def __init__(self,
                 observation_space: gym.Space,
                 action_space: gym.Space,
                 num_actors=1,
                 log_time_interval: float=None,
                 disable_training=False):
        """
        Base class for all reinforcement learning algorithms. Supports running parallely on multiple envs.
        Args:
            observation_space: Env observation space.
            action_space: Env action space.
            log_time_interval: Logging interval in seconds. None disables logging.
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.num_actors = num_actors
        self.disable_training = disable_training
        self.step_type = RLStep.EVAL
        self.log_time_interval = log_time_interval

        self.cur_states, self.prev_states, self.rewards, self.dones = [None] * 4
        self._logger = None
        self._last_log_time = 0
        self._do_log = False
        self.step = 0

    @property
    def frame(self):
        """Processed frames across all actors"""
        return self.step * self.num_actors

    @property
    def logger(self):
        """
        Return logging class. Could be of any type, but only `TensorboardEnvLogger` is currently used.
        None if logging is disabled.
        """
        return self._logger

    @logger.setter
    def logger(self, log):
        self._logger = log
        self._log_set()

    def _step(self, prev_states: np.ndarray, rewardss: np.ndarray,
              doness: np.ndarray, cur_states: np.ndarray) -> np.ndarray:
        """
        Internal RL algorithm step.
        Args:
            prev_states: Previous observations.
            rewards: Rewards received after actig on `prev_states`
            dones: Episode end flags.
            cur_states: Current observations.

        Returns: Actions for `current_states`
        """
        raise NotImplementedError

    def eval(self, input: np.ndarray or list) -> np.ndarray:
        """
        Process new observations and return actions.
        Args:
            input: List of observations across all `envs`
            envs: List of parallely running envs.

        Returns: Taken actions.
        """
        self.prev_states = self.cur_states
        self.cur_states = self._check_states(input)
        actions = self._step(self.prev_states, self.rewards, self.dones, self.cur_states)
        self.step += 1
        if actions is None:
            return None
        if isinstance(self.action_space, Discrete):
            actions = np.reshape(actions, (self.num_actors,))
        else:
            actions = np.reshape(actions, (self.num_actors, -1))
        return actions

    def reward(self, reward: np.ndarray or list) -> None:
        """
        Reward for taken actions at `self.eval` call.
        Args:
            reward: Rewards
        """
        self.rewards = self._check_rewards(reward)

    def finish_episodes(self, done: np.ndarray or list) -> None:
        """
        Notify for ended episodes after taking actions from `self.eval`.
        Args:
            done: Episode end flags
        """
        self.dones = self._check_dones(done)

    def _log_set(self):
        """Called when logger is set or changed"""
        pass

    def _check_states(self, input) -> np.ndarray:
        """
        Check if observations have correct shape and type and convert them to numpy array.
            Also check if it's allowed to call that function in current `self.step_type`
        Args:
            input: Observations

        Returns: Observations converted to numpy array
        """
        assert self.step_type == RLStep.EVAL or self.disable_training
        input = np.asarray(input, dtype=np.float32)
        assert input.shape[1:] == self.observation_space.shape, f'{input.shape[1:]} {self.observation_space.shape}'
        self.step_type = RLStep.REWARD
        return input

    def _check_rewards(self, rewards):
        """
        Check if rewards have correct shape and type and convert them to numpy array.
            Also check if it's allowed to call that function in current `self.step_type`
        Args:
            rewards: Rewards

        Returns: Rewards converted to numpy array
        """
        assert self.step_type == RLStep.REWARD
        rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
        assert rewards.shape == (self.num_actors,), f'wrong reward {rewards} shape {rewards.shape}'
        self.step_type = RLStep.FINISH
        return rewards

    def _check_dones(self, done):
        """
        Check if done flags have correct shape and type and convert them to numpy array.
            Also check if it's allowed to call that function in current `self.step_type`
        Args:
            done: Episode end flags

        Returns: Episode end flags converted to numpy array
        """
        assert self.step_type == RLStep.FINISH
        done = np.asarray(done, dtype=bool).reshape(-1)
        assert done.shape == (self.num_actors,)
        self.step_type = RLStep.EVAL
        return done

    def _check_log(self):
        """Check if logging should be enabled for current step."""
        if self.logger is not None and self.log_time_interval is not None and \
                                self._last_log_time + self.log_time_interval < time.time():
            self._last_log_time = time.time()
            self._do_log = True
        else:
            self._do_log = False