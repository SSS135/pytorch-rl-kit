from enum import Enum
from typing import Optional, NamedTuple

import gymnasium as gym
import torch
from gymnasium.spaces import Discrete
from .model_saver import ModelSaver

if __name__ != '__main__':
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.jit.enable_onednn_fusion(True)


class RLStepData(NamedTuple):
    rewards: torch.Tensor
    true_reward: torch.Tensor
    done: torch.Tensor
    obs: torch.Tensor
    actor_id: torch.Tensor


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
                 actor_index=0,
                 num_rewards=1):
        """
        Base class for all reinforcement learning algorithms. Supports running parallely on multiple envs.
        Args:
            observation_space: Env observation space.
            action_space: Env action space.
            log_interval: Logging interval in frames. None disables logging.
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.num_actors = None if self.has_variable_actor_count_support else num_actors
        self.log_interval = log_interval
        self.disable_training = disable_training
        self.actor_index = actor_index
        self.model_save_folder = model_save_folder
        self.model_save_interval = model_save_interval
        self.save_intermediate_models = save_intermediate_models
        self.model_save_tag = model_save_tag
        self.model_init_path = model_init_path
        self.num_rewards = num_rewards

        self._logger = None
        self._last_log_frame = -log_interval if log_interval is not None else None
        self._do_log = False
        self.frame_eval = 0
        self.frame_train = 0

        self._model_saver = ModelSaver(model_save_folder, model_save_tag, model_save_interval,
                                       save_intermediate_models, self.actor_index)

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

    @property
    def has_variable_actor_count_support(self):
        return False

    def _step(self, data: RLStepData) -> torch.Tensor:
        raise NotImplementedError

    def step(self, obs: torch.Tensor, rewards: torch.Tensor, done: torch.Tensor,
             true_reward: torch.Tensor, actor_id: torch.Tensor) -> torch.Tensor:
        data = self._preprocess_data(obs=obs, rewards=rewards, true_reward=true_reward,
                                     done=done, actor_id=actor_id)
        actions = self._step(data)
        self.frame_eval += data.obs.shape[0]
        if isinstance(self.action_space, Discrete):
            return actions.reshape(actions.shape[0])
        else:
            return actions.reshape(actions.shape[0], -1)

    def _preprocess_data(self, obs: torch.Tensor, rewards: torch.Tensor, done: torch.Tensor,
                         true_reward: torch.Tensor, actor_id: torch.Tensor) -> RLStepData:
        num_actors = obs.shape[0]

        if rewards.ndim == 1:
            rewards = rewards.unsqueeze(-1)
        if true_reward is None:
            true_reward = rewards[:, 0]
        if actor_id is None:
            actor_id = torch.arange(num_actors, dtype=torch.long)
        if not self.has_variable_actor_count_support:
            assert torch.allclose(actor_id, torch.arange(self.num_actors, dtype=actor_id.dtype)), (actor_id, self.num_actors)
        if obs.dtype == torch.float64:
            obs = obs.float()

        assert obs.shape == (num_actors, *self.observation_space.shape), f'{obs.shape} {self.observation_space.shape}'
        assert obs.dtype in (torch.float32, torch.uint8), obs.dtype
        assert rewards.shape == (num_actors, rewards.shape[1]), f'wrong reward {rewards} shape {rewards.shape}'
        assert true_reward.shape == (num_actors,)
        assert done.shape == (num_actors,)
        assert actor_id.shape == (num_actors,)
        assert actor_id.dtype == torch.long or actor_id.dtype == torch.int, actor_id.dtype
        aid_list = actor_id.tolist()
        assert len(aid_list) == len(set(aid_list)), aid_list

        rewards = rewards.float()
        true_reward = true_reward.float()
        done = done.float()
        actor_id = actor_id.long()

        return RLStepData(obs=obs, rewards=rewards, true_reward=true_reward, done=done, actor_id=actor_id)

    def drop_collected_steps(self):
        pass

    def _log_set(self):
        """Called when logger is set or changed"""
        pass

    def _check_log(self):
        """Check if logging should be enabled for current step."""
        if self.logger is not None and self.log_interval is not None and \
                self.frame_train >= self._last_log_frame + self.log_interval:
            self._last_log_frame += self.log_interval
            self._do_log = True
        else:
            self._do_log = False
