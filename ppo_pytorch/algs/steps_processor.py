import random
from typing import Dict, Optional

from ..common.attr_dict import AttrDict
from ..common.barron_loss import barron_loss_derivative
from ..common.gae import calc_advantages, calc_value_targets, calc_advantages_noreward
import torch


class StepsProcessor:
    def __init__(self,
                 pd,
                 reward_discount,
                 advantage_discount,
                 reward_scale,
                 mean_norm,
                 barron_alpha_c,
                 entropy_reward_scale,
                 prev_steps_processor):
        super().__init__()
        self.pd = pd
        self.reward_discount = reward_discount
        self.advantage_discount = advantage_discount
        self.reward_scale = reward_scale
        self.mean_norm = mean_norm
        self.barron_alpha_c = barron_alpha_c
        self.entropy_reward_scale = entropy_reward_scale
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

        self.data.rewards += self._get_entropy_rewards()

        self.data.rewards, self.data.state_value_targets, self.data.advantages = \
            self._process_rewards(self.data.rewards, self.data.state_values, self.data.dones)

        self._drop_excess_data()
        self._flatten_data()

    def _data_to_tensors(self):
        self.data.update({k: self._to_tensor(v) for k, v in self.data.items()})

    def _get_entropy_rewards(self):
        if self.entropy_reward_scale != 0:
            entropy = self.pd.entropy(self.data.logits[:-1]).mean(-1) * self.data.rewards.pow(2).mean().sqrt()
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

    def _process_rewards(self, rewards, values, dones, barron_scale=True):
        norm_rewards = self.reward_scale * rewards

        # calculate state_value_targets and advantages
        state_value_targets = calc_value_targets(norm_rewards, values, dones, self.reward_discount, self.reward_discount)
        advantages = calc_advantages(norm_rewards, values, dones, self.reward_discount, self.advantage_discount)

        advantages = self._normalize_advantages(advantages, barron_scale)

        return norm_rewards, state_value_targets, advantages

    def _normalize_advantages(self, advantages, barron_scale):
        return (advantages - advantages.mean()) / max(advantages.std(), 1e-4)
        # mean, square, iter = self._advantage_stats
        # mean = self._advantage_momentum * mean + (1 - self._advantage_momentum) * advantages.mean().item()
        # square = self._advantage_momentum * square + (1 - self._advantage_momentum) * advantages.pow(2).mean().item()
        # iter += 1
        # self._advantage_stats = (mean, square, iter)
        #
        # bias_corr = 1 - self._advantage_momentum ** iter
        # mean = mean / bias_corr
        # square = square / bias_corr
        #
        # if self.mean_norm:
        #     std = (square - mean ** 2) ** 0.5
        #     advantages = (advantages - mean) / max(std, 1e-3)
        # else:
        #     rms = square ** 0.5
        #     advantages = advantages / max(rms, 1e-3)

        # def adv_norm(advantages):
        #     if self.mean_norm:
        #         return (advantages - advantages.mean()) / max(advantages.std(), 1e-4)
        #     else:
        #         return advantages / max(advantages.pow(2).mean().sqrt(), 1e-4)
        #
        # advantages = adv_norm(advantages)
        # if barron_scale:
        #     advantages = barron_loss_derivative(advantages, *self.barron_alpha_c)
        #     advantages = adv_norm(advantages)
        # advantages.clamp_(-10, 10)

        # adv_shape = advantages.shape
        # advantages = advantages.view(-1)
        # advantages[advantages.argsort()] = torch.linspace(
        #     -1.7062, 1.7062, advantages.shape[0], dtype=advantages.dtype, device=advantages.device)
        # advantages = advantages.view(adv_shape)

        # return advantages