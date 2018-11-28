from typing import Dict

from ..common.attr_dict import AttrDict
from ..common.barron_loss import barron_loss_derivative
from ..common.gae import calc_returns, calc_advantages
import torch


class StepsProcessor:
    def __init__(self,
                 pd,
                 reward_discount,
                 advantage_discount,
                 reward_scale,
                 mean_norm,
                 barron_alpha_c,
                 entropy_reward_scale):
        super().__init__()
        self.pd = pd
        self.reward_discount = reward_discount
        self.advantage_discount = advantage_discount
        self.reward_scale = reward_scale
        self.mean_norm = mean_norm
        self.barron_alpha_c = barron_alpha_c
        self.entropy_reward_scale = entropy_reward_scale
        self.data = AttrDict()

    def append_head(self, head: Dict[str, torch.Tensor], **new_data: torch.Tensor):
        self.append_values(**head, **new_data)

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

        self.data.rewards += self._get_entropy_rewards()

        self.data.rewards, self.data.returns, self.data.advantages = \
            self._process_rewards(self.data.rewards, self.data.state_values, self.data.dones)

        self._drop_excess_data()
        self._flatten_data()

    def _data_to_tensors(self):
        self.data.update({k: self._to_tensor(v) for k, v in self.data.items()})

    def _get_entropy_rewards(self):
        if self.entropy_reward_scale != 0:
            entropy = self.pd.entropy(self.data.probs[:-1]).mean(-1) * self.data.rewards.pow(2).mean().sqrt()
            return self.entropy_reward_scale * entropy
        else:
            return self.data.rewards.new_zeros(1)

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

        # calculate returns and advantages
        returns = calc_returns(norm_rewards, values, dones, self.reward_discount)
        advantages = calc_advantages(norm_rewards, values, dones, self.reward_discount, self.advantage_discount)
        if self.mean_norm:
            advantages = (advantages - advantages.mean()) / max(advantages.std(), 1e-3)
        else:
            advantages = advantages / max(advantages.pow(2).mean().sqrt(), 1e-3)
        advantages = barron_loss_derivative(advantages, *self.barron_alpha_c)

        return norm_rewards, returns, advantages