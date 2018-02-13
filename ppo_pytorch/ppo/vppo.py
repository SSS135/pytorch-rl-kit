import pprint
from collections import namedtuple, deque
from functools import partial
import copy
import math

import gym.spaces
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from ..common import DecayLR, ValueDecay
from ..common.gae import calc_advantages, calc_returns
from ..common.multi_dataset import MultiDataset
from ..common.probability_distributions import DiagGaussianPd
from ..common.rl_base import RLBase
from ..models import MLPActorCritic
from .ppo import PPO, TrainingData


class VPPO(PPO):
    def __init__(self, observation_space, action_space, *args,
                 validation_horizon=16,
                 horizon=64,
                 error_threshold=90,
                 **kwargs):
        """
        Args:
            observation_space (gym.Space): Environment's observation space
            action_space (gym.Space): Environment's action space
        """
        super().__init__(observation_space, action_space, *args,
                         horizon=horizon + validation_horizon,
                         **kwargs)
        assert self.constraint == 'clip_mod'
        self.validation_horizon = validation_horizon
        self.error_threshold = error_threshold
        self._value_errors = deque(maxlen=100)
        self._policy_errors = deque(maxlen=100)
        self._passed_statistics = 0.5

    def _train(self):
        all_data = self._prepare_training_data()
        train_data, val_data = self._train_val_split(all_data)

        self.model.set_log(None, False, self.step)
        model_copy = copy.deepcopy(self.model)

        self._ppo_update(train_data)

        if not self._validate(val_data):
            for (src, dst) in zip(model_copy.parameters(), self.model.parameters()):
                dst.data.copy_(src.data)

    def _validate(self, data: TrainingData):
        val_head = self.model(Variable(data.states, volatile=True))
        values = val_head.state_values.data.cpu()
        probs = val_head.probs.data.cpu()

        policy_clip = self.policy_clip * self.clip_mult
        value_clip = self.value_clip * self.clip_mult

        logp = self.model.pd.logp(data.actions, probs)
        logp_old = self.model.pd.logp(data.actions, data.probs_old)
        ratio = (logp - logp_old).exp()

        ratio_target = data.advantages.sign().mul_(policy_clip).exp_()
        cur_policy_rmse = (ratio - ratio_target).abs().mean()
        cur_policy_baseline_err = (1 - ratio_target).abs().mean()
        policy_err = cur_policy_rmse / cur_policy_baseline_err

        value_target = data.values_old + (data.returns - data.values_old).clamp(-value_clip, value_clip)
        cur_value_rmse = (values - value_target).abs().mean()
        cur_value_baseline = (data.values_old - value_target).abs().mean()

        value_err = cur_value_rmse / cur_value_baseline

        if len(self._policy_errors) > 20:
            policy_threshold = np.percentile(self._policy_errors, self.error_threshold)
            value_threshold = np.percentile(self._value_errors, self.error_threshold)
        else:
            policy_threshold = 1.1 * policy_err
            value_threshold = 1.1 * value_err

        passed = value_err < value_threshold and policy_err < policy_threshold

        self._passed_statistics = 0.99 * self._passed_statistics + 0.01 * passed

        self._policy_errors.append(policy_err)
        self._value_errors.append(value_err)

        if self._do_log:
            self.logger.add_scalar('policy validation error', policy_err, self.frame)
            self.logger.add_scalar('value validation error', value_err, self.frame)
            self.logger.add_scalar('value validation error threshold', value_threshold, self.frame)
            self.logger.add_scalar('policy validation error threshold', policy_threshold, self.frame)
            self.logger.add_scalar('passed ppo updates fraction', self._passed_statistics, self.frame)

        return passed

    def _train_val_split(self, all_data) -> (TrainingData, TrainingData):
        def stepwise_slice(input, indexes):
            x = input.view(-1, self.num_actors, *input.shape[1:])
            x = x[indexes.cuda() if x.is_cuda else indexes].contiguous()
            return x.view(x.shape[0] * x.shape[1], *x.shape[2:])

        all_perm = torch.randperm(self.horizon - 1)

        all_data = all_data._asdict()
        train_horizon = self.horizon - self.validation_horizon
        train_data = TrainingData._make([stepwise_slice(v, all_perm[:train_horizon]) for v in all_data.values()])
        val_data = TrainingData._make([stepwise_slice(v, all_perm[train_horizon:]) for v in all_data.values()])
        return train_data, val_data
