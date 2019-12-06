from enum import Enum
from typing import Optional

import gym
import torch
from gym.spaces import Discrete
from .model_saver import ModelSaver


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
                 log_interval: float=None,
                 disable_training=False,
                 model_save_folder=None,
                 model_save_tag=None,
                 model_save_interval=100_000,
                 model_init_path=None,
                 save_intermediate_models=False,
                 actor_index=0,):
        """
        Base class for all reinforcement learning algorithms. Supports running parallely on multiple envs.
        Args:
            observation_space: Env observation space.
            action_space: Env action space.
            log_interval: Logging interval in frames. None disables logging.
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.num_actors = num_actors
        self.log_interval = log_interval
        self.disable_training = disable_training
        self.actor_index = actor_index
        self.model_save_folder = model_save_folder
        self.model_save_interval = model_save_interval
        self.save_intermediate_models = save_intermediate_models
        self.model_save_tag = model_save_tag
        self.model_init_path = model_init_path

        self.step_type = RLStep.EVAL
        self.cur_states, self.prev_states, self.rewards, self.dones = [None] * 4
        self._logger = None
        self._last_log_frame = -log_interval
        self._do_log = False
        self.step_eval = 0
        self.step_train = 0

        self._model_saver = ModelSaver(model_save_folder, model_save_tag, model_save_interval,
                                       save_intermediate_models, self.actor_index)

    @property
    def frame_eval(self):
        """Processed frames across all actors"""
        return self.step_eval * self.num_actors

    @property
    def frame_train(self):
        """Processed frames across all actors"""
        return self.step_train * self.num_actors

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

    def _step(self, rewards: torch.Tensor, dones: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """
        Internal RL algorithm step.
        Args:
            rewards: Rewards received after actig on `prev_states`
            dones: Episode end flags.
            states: Current observations.

        Returns: Actions for `current_states`
        """
        raise NotImplementedError

    def eval(self, obs: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Process new observations and return actions.
        Args:
            obs: List of observations across all `envs`
            envs: List of parallely running envs.

        Returns: Taken actions.
        """
        self.prev_states = self.cur_states
        self.cur_states = self._check_states(obs)
        actions = self._step(self.rewards, self.dones, self.cur_states)
        self.step_eval += 1
        if actions is None:
            return None
        if isinstance(self.action_space, Discrete):
            actions = actions.reshape(self.num_actors)
        else:
            actions = actions.reshape(self.num_actors, -1)
        return actions

    def reward(self, reward: torch.Tensor):
        """
        Reward for taken actions at `self.eval` call.
        Args:
            reward: Rewards
        """
        self.rewards = self._check_rewards(reward)

    def finish_episodes(self, done: torch.Tensor):
        """
        Notify for ended episodes after taking actions from `self.eval`.
        Args:
            done: Episode end flags
        """
        self.dones = self._check_dones(done)

    def drop_collected_steps(self):
        pass

    def _log_set(self):
        """Called when logger is set or changed"""
        pass

    def _check_states(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Check if observations have correct shape and type and convert them to numpy array.
            Also check if it's allowed to call that function in current `self.step_type`
        Args:
            obs: Observations

        Returns: Observations converted to numpy array
        """
        assert self.step_type == RLStep.EVAL or self.disable_training
        assert obs.shape[1:] == self.observation_space.shape, f'{obs.shape[1:]} {self.observation_space.shape}'
        assert obs.dtype == torch.float32 or obs.dtype == torch.uint8
        self.step_type = RLStep.REWARD
        return obs

    def _check_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Check if rewards have correct shape and type and convert them to numpy array.
            Also check if it's allowed to call that function in current `self.step_type`
        Args:
            rewards: Rewards

        Returns: Rewards converted to numpy array
        """
        assert self.step_type == RLStep.REWARD
        assert rewards.shape == (self.num_actors,), f'wrong reward {rewards} shape {rewards.shape}'
        self.step_type = RLStep.FINISH
        return rewards

    def _check_dones(self, done: torch.Tensor) -> torch.Tensor:
        """
        Check if done flags have correct shape and type and convert them to numpy array.
            Also check if it's allowed to call that function in current `self.step_type`
        Args:
            done: Episode end flags

        Returns: Episode end flags converted to numpy array
        """
        assert self.step_type == RLStep.FINISH
        assert done.shape == (self.num_actors,)
        self.step_type = RLStep.EVAL
        return done

    def _check_log(self):
        """Check if logging should be enabled for current step."""
        if self.logger is not None and self.log_interval is not None and \
                self.frame_train >= self._last_log_frame + self.log_interval:
            self._last_log_frame += self.log_interval
            self._do_log = True
        else:
            self._do_log = False
