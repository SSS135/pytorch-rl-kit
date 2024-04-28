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
                 minmax_perc=(0.05, 0.95),
                 perc_decay=0.99,
                 perc_lowhigh=None):
        super().__init__()
        self.pd = pd
        self.reward_discount = reward_discount
        self.advantage_discount = advantage_discount
        self.minmax_perc = minmax_perc
        self.perc_decay = perc_decay
        self.perc_lowhigh = perc_lowhigh or torch.tensor([0, 1])
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
        state_value_targets = calc_value_targets(rewards, values, dones, self.reward_discount, self.reward_discount)
        advantages = calc_advantages(rewards, values, dones, self.reward_discount, self.advantage_discount)
        return rewards, state_value_targets, advantages
