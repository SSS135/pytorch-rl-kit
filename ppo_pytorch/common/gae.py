import math
from typing import Tuple, Optional

import torch
from torch import Tensor as TT


def _check_data(rewards: TT, values: TT, dones: TT):
    assert len(rewards) == len(dones) == len(values) - 1, (rewards.shape, dones.shape, values.shape)
    # (steps, actors)
    assert rewards.dim() == dones.dim() == 2, (rewards.shape, dones.shape, values.shape)
    # (steps, actors)
    assert values.dim() == 2, (rewards.shape, dones.shape, values.shape)


def calc_advantages(rewards: TT, values: TT, dones: TT, reward_discount: float, advantage_discount: float) -> TT:
    """
    Calculate advantages with Global Advantage Estimation
    Args:
        rewards: Rewards from environment
        values: State-values
        dones: Episode ended flags
        reward_discount: Discount factor for state-values
        advantage_discount: Discount factor for advantages

    Returns: Advantages with GAE
    """
    _check_data(rewards, values, dones)

    next_adv = 0
    advantages = torch.zeros_like(rewards)
    for t in reversed(range(len(rewards))):
        nonterminal = 1 - dones[t]
        td_residual = rewards[t] + reward_discount * nonterminal * values[t + 1] - values[t]
        advantages[t] = next_adv = td_residual + advantage_discount * reward_discount * nonterminal * next_adv

    return advantages


def calc_advantages_noreward(rewards: TT, values: TT, dones: TT, reward_discount: float, advantage_discount: float) -> TT:
    _check_data(rewards, values, dones)

    targets = values.clone()
    for t in reversed(range(len(dones))):
        done = dones[t]
        target_no_done = values[t] * (1 - reward_discount) + \
                         targets[t + 1] * reward_discount * advantage_discount + \
                         values[t + 1] * reward_discount * (1 - advantage_discount)
        targets[t] = done * rewards[t] + (1 - done) * target_no_done

    return targets[:-1] - values[:-1]


def calc_value_targets(rewards: TT, values: TT, dones: TT, reward_discount: float, gae_lambda=1.0) -> TT:
    """
    Calculate temporal difference targets
    Args:
        rewards: Rewards from environment (steps, actors)
        values: State-values (steps, actors)
        dones: Episode ended flags (steps, actors)
        reward_discount: Discount factor for state-values

    Returns: Target values (steps, actors)
    """
    _check_data(rewards, values, dones)

    R = values[-1]
    targets = rewards.new_zeros((values.shape[0] - 1, *values.shape[1:]))
    for t in reversed(range(len(rewards))):
        nonterminal = 1 - dones[t]
        R = rewards[t] + nonterminal * reward_discount * R
        targets[t] = R
        R = (1 - gae_lambda) * values[t] + gae_lambda * R

    return targets


def calc_vtrace(rewards: TT, values: TT, dones: TT, probs_ratio: TT, discount: float, c_max=1.5, p_max=1.5) -> Tuple[TT, TT, TT]:
    _check_data(rewards, values, dones)
    assert probs_ratio.shape == rewards.shape, (probs_ratio.shape, rewards.shape)

    c = probs_ratio.clamp(0, c_max)
    p = probs_ratio.clamp(0, p_max)
    nonterminal = 1 - dones
    td = p * (rewards + nonterminal * discount * values[1:] - values[:-1])
    targets = values.clone()
    for i in reversed(range(len(rewards))):
        targets[i] = values[i] + td[i] + nonterminal[i] * discount * c[i] * (targets[i + 1] - values[i + 1])
    advantages = rewards + nonterminal * discount * targets[1:] - values[:-1]

    assert targets.shape == values.shape, (targets.shape, values.shape)
    assert advantages.shape == rewards.shape, (advantages.shape, rewards.shape)

    return targets[:-1], advantages.mean(-1).mean(-1), p.mean(-1).mean(-1)


def assert_equal_tensors(a, b, abs_tol=1e-4):
    assert a.shape == b.shape, (a.shape, b.shape)
    assert (a - b).abs().max().item() < abs_tol, (a, b)


def test_vtrace():
    N = (1000, 8)
    discount = 0.99
    rewards = torch.randn(N)
    values = torch.randn((N[0] + 1, N[1]))
    dones = (torch.rand(N) > 0.95).float()
    cur_probs = old_probs = torch.zeros(N)
    ret = calc_value_targets(rewards, values, dones, discount)
    adv = calc_advantages(rewards, values, dones, discount, 1)
    v_ret, v_adv = calc_vtrace(rewards, values, dones, cur_probs / old_probs, discount, 1, 1)

    assert ((ret - v_ret).abs() > 1e-2).sum().item() == 0
    assert ((adv - v_adv).abs() > 1e-2).sum().item() == 0
