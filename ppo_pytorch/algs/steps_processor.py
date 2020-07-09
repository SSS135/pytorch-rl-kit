import random
from typing import Dict, Optional

from ..common.attr_dict import AttrDict
from ..common.gae import calc_advantages, calc_value_targets, calc_advantages_noreward
import torch

from ..common.squash import unsquash, squash


class StepsProcessor:
    def __init__(self,
                 pd,
                 reward_discount,
                 advantage_discount,
                 reward_scale,
                 mean_norm,
                 squash_values):
        super().__init__()
        self.pd = pd
        self.reward_discount = reward_discount
        self.advantage_discount = advantage_discount
        self.reward_scale = reward_scale
        self.mean_norm = mean_norm
        self.squash_values = squash_values
        self.data = AttrDict()

    def append_values(self, **new_data: torch.Tensor):
        first_step = 'actions' not in self.data
        for k, v in new_data.items():
            if (k == 'rewards' or k == 'dones') and first_step:
                continue
            if k not in self.data:
                self.data[k] = []
            self.data[k].append(v.cpu())

    def complete(self):
        self._data_to_tensors()

        self.data.rewards, self.data.state_value_targets, self.data.advantages = \
            self._process_rewards(self.data.rewards, self.data.state_values, self.data.dones)

        self._drop_excess_data()
        self._flatten_data()

    def _data_to_tensors(self):
        self.data.update({k: self._to_tensor(v) for k, v in self.data.items()})

    def _drop_excess_data(self):
        lens = [len(x) for x in self.data.values()]
        min_len = min(lens)
        self.data.update({k: v[:min_len] for k, v in self.data.items()})

    def _flatten_data(self):
        self.data.update({k: v.reshape(-1, *v.shape[2:]) for k, v in self.data.items()})

    def _to_tensor(self, values):
        return values if torch.is_tensor(values) else torch.stack(values, dim=0)

    def _process_rewards(self, rewards, values, dones):
        norm_rewards = self.reward_scale * rewards

        if self.squash_values:
            values = unsquash(values)
        state_value_targets = calc_value_targets(norm_rewards, values, dones, self.reward_discount, self.reward_discount)
        advantages = calc_advantages(norm_rewards, values, dones, self.reward_discount, self.advantage_discount)
        advantages = self._normalize_advantages(advantages)
        if self.squash_values:
            state_value_targets = squash(state_value_targets)

        return norm_rewards, state_value_targets, advantages

    def _normalize_advantages(self, advantages):
        return (advantages - advantages.mean()) / max(advantages.std(), 1e-6)