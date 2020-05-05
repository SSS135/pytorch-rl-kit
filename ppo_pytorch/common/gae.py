import random

import math
from typing import Tuple, Optional

import torch
import torch.jit


def _check_data(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor):
    assert len(rewards) == len(dones) == len(values) - 1, (rewards.shape, dones.shape, values.shape)
    # (steps, actors)
    assert rewards.dim() == dones.dim() == 2, (rewards.shape, dones.shape, values.shape)
    # (steps, actors)
    assert values.dim() == 2, (rewards.shape, dones.shape, values.shape)


@torch.jit.script
def calc_advantages(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor,
                    reward_discount: float, advantage_discount: float) -> torch.Tensor:
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

    advantages = torch.zeros_like(rewards)
    next_adv = torch.zeros_like(advantages[0])
    for t_inv in range(rewards.shape[0]):
        t = rewards.shape[0] - 1 - t_inv
        nonterminal = 1 - dones[t]
        td_residual = rewards[t] + reward_discount * nonterminal * values[t + 1] - values[t]
        advantages[t] = next_adv = td_residual + advantage_discount * reward_discount * nonterminal * next_adv

    return advantages


@torch.jit.script
def calc_advantages_noreward(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor,
                             reward_discount: float, advantage_discount: float) -> torch.Tensor:
    _check_data(rewards, values, dones)

    targets = values.clone()
    for t_inv in range(rewards.shape[0]):
        t = rewards.shape[0] - 1 - t_inv
        done = dones[t]
        target_no_done = values[t] * (1 - reward_discount) + \
                         targets[t + 1] * reward_discount * advantage_discount + \
                         values[t + 1] * reward_discount * (1 - advantage_discount)
        targets[t] = done * rewards[t] + (1 - done) * target_no_done

    return targets[:-1] - values[:-1]


@torch.jit.script
def calc_value_targets(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor,
                       reward_discount: float, gae_lambda: float = 1.0, upgo: bool = False) -> torch.Tensor:
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
    targets_shape = list(values.shape)
    targets_shape[0] = targets_shape[0] - 1
    targets = torch.zeros(targets_shape, device=values.device, dtype=values.dtype)
    one = torch.scalar_tensor(1, device=values.device, dtype=values.dtype)
    nonterminal = 1 - dones
    for t_inv in range(rewards.shape[0]):
        t = rewards.shape[0] - 1 - t_inv
        if upgo and t + 1 < rewards.shape[0]:
            good = rewards[t + 1].addcmul(nonterminal[t + 1], values[t + 2], value=reward_discount).gt_(values[t + 1])
        else:
            good = one
        rewards[t].addcmul(nonterminal[t], torch.lerp(values[t + 1], R, gae_lambda * good), value=reward_discount, out=targets[t])
        R = targets[t]

    return targets


@torch.jit.script
def calc_vtrace(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor,
                probs_ratio: torch.Tensor, kl_div: torch.Tensor,
                discount: float, max_ratio: float = 1.0, kl_limit: float = 0.3
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _check_data(rewards, values, dones)
    assert probs_ratio.shape == rewards.shape == kl_div.shape == dones.shape, (probs_ratio.shape, rewards.shape, kl_div.shape)
    assert rewards.shape[0] == values.shape[0] - 1 and rewards.shape[1:] == values.shape[1:]

    p = probs_ratio.clamp_max(max_ratio) * (kl_div < kl_limit).float()
    nonterminal = 1 - dones

    value_targets = values.clone()
    for i_inv in range(rewards.shape[0]):
        i = rewards.shape[0] - 1 - i_inv
        value_targets[i] = torch.lerp(values[i], rewards[i] + discount * nonterminal[i] * value_targets[i + 1], p[i])

    advantages_vtrace = value_targets[:-1] - values[:-1]
    advantages_upgo = calc_value_targets(rewards, values, dones, discount, upgo=True) - values[:-1]

    assert value_targets.shape == values.shape, (value_targets.shape, values.shape)
    assert advantages_vtrace.shape == advantages_upgo.shape == rewards.shape, (advantages_vtrace.shape, rewards.shape)

    return value_targets[:-1], advantages_vtrace, advantages_upgo, p


@torch.jit.script
def calc_vtrace_old(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor,
                probs_ratio: torch.Tensor, kl_div: torch.Tensor,
                discount: float, max_ratio: float = 1.0, kl_limit: float = 0.3
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _check_data(rewards, values, dones)
    assert probs_ratio.shape == rewards.shape == kl_div.shape == dones.shape, (probs_ratio.shape, rewards.shape, kl_div.shape)
    assert rewards.shape[0] == values.shape[0] - 1 and rewards.shape[1:] == values.shape[1:]

    # if max_ratio > 1.0:
    #     for i_inv in range(probs_ratio.shape[0] - 1):
    #         i = probs_ratio.shape[0] - 2 - i_inv
    #         probs_ratio[i] *= torch.addcmul(dones[i], probs_ratio[i + 1].clamp(1.0, max_ratio), nonterminal[i])
    kl_mask = 1.0 - (kl_div / kl_limit).clamp_max(1.0)
    p = (probs_ratio.clamp_max(max_ratio) * kl_mask).clamp_max(1.0)
    # dones = p * dones
    nonterminal = 1 - dones
    deltas = p * (rewards + nonterminal * discount * values[1:] - values[:-1])
    nonterm_c = nonterminal * p
    vs_minus_v_xs = torch.zeros_like(values)
    for i_inv in range(rewards.shape[0]):
        i = rewards.shape[0] - 1 - i_inv
        # vs_minus_v_xs[i] = deltas[i] + discount * torch.lerp(c[i] * vs_minus_v_xs[i + 1], 0, dones[i])
        torch.addcmul(deltas[i], nonterm_c[i], vs_minus_v_xs[i + 1], value=discount, out=vs_minus_v_xs[i])
    value_targets = vs_minus_v_xs + values
    advantages_vtrace = rewards + nonterminal * discount * value_targets[1:] - values[:-1]
    advantages_upgo = calc_value_targets(rewards, values, dones, discount, upgo=True) - values[:-1]

    assert value_targets.shape == values.shape, (value_targets.shape, values.shape)
    assert advantages_vtrace.shape == advantages_upgo.shape == rewards.shape, (advantages_vtrace.shape, rewards.shape)

    return value_targets[:-1], advantages_vtrace, advantages_upgo, p


def assert_equal_tensors(a, b, abs_tol=1e-4):
    assert a.shape == b.shape, (a.shape, b.shape)
    assert (a - b).abs().max().item() < abs_tol, (a, b)


def test_vtrace():
    torch.manual_seed(123)
    N = (1000, 8)
    discount = 0.99
    rewards = torch.randn(N)
    values = torch.randn((N[0] + 1, N[1]))
    dones = (torch.rand(N) > 0.95).float()
    prob_ratio = torch.ones(N)
    kl_div = torch.zeros(N)
    ret = calc_value_targets(rewards, values, dones, discount)
    adv = calc_advantages(rewards, values, dones, discount, 1)
    v_ret, v_adv, upgo, p = calc_vtrace(rewards, values, dones, prob_ratio, kl_div, discount, 2.0, 0.3)

    assert v_ret.shape == ret.shape
    assert v_adv.shape == adv.shape
    assert ((ret - v_ret).abs() > 1e-3).sum().item() == 0, (ret[-8:, 0], v_ret[-8:, 0])
    assert ((adv - v_adv).abs() > 1e-3).sum().item() == 0, (adv[-8:, 0], v_adv[-8:, 0])
