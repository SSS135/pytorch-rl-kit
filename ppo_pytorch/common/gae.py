import random

import math
from typing import Tuple, Optional

import torch
import torch.jit
from torch import Tensor


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
                       reward_discount: float, gae_lambda: float = 1.0) -> torch.Tensor:
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

    nonterminal = 1 - dones
    targets = values.clone()

    for t_inv in range(rewards.shape[0]):
        t = rewards.shape[0] - 1 - t_inv
        targets[t] = rewards[t] + nonterminal[t] * reward_discount * torch.lerp(values[t + 1], targets[t + 1], gae_lambda)

    return targets[:-1]


@torch.jit.script
def calc_upgo(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor,
              reward_discount: float, gae_lambda: float = 1.0, q_values: Optional[Tensor]=None) -> torch.Tensor:
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

    nonterm_disc = reward_discount * (1 - dones)
    targets = values.clone()
    if q_values is None:
        q_values = rewards + nonterm_disc * values[1:]
    else:
        assert values.shape == q_values.shape
        q_values = q_values[:-1]
    target_factor = gae_lambda * (q_values > values[:-1]).float()

    for t_inv in range(rewards.shape[0]):
        t = rewards.shape[0] - 1 - t_inv
        lerp = target_factor[t + 1] if t_inv > 0 else target_factor[t]
        targets[t] = rewards[t] + nonterm_disc[t] * torch.lerp(values[t + 1], targets[t + 1], lerp)

    return targets[:-1]


@torch.jit.script
def calc_vtrace(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor,
                probs_ratio: torch.Tensor, kl_div: torch.Tensor,
                discount: float, max_ratio: float = 2.0, kl_limit: float = 0.3
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _check_data(rewards, values, dones)
    assert probs_ratio.shape == rewards.shape == kl_div.shape == dones.shape, (probs_ratio.shape, rewards.shape, kl_div.shape)
    assert rewards.shape[0] == values.shape[0] - 1 and rewards.shape[1:] == values.shape[1:]

    nonterminal = 1 - dones
    if max_ratio > 1.0:
        for i_inv in range(probs_ratio.shape[0] - 1):
            i = probs_ratio.shape[0] - 2 - i_inv
            probs_ratio[i] *= torch.addcmul(dones[i], probs_ratio[i + 1].clamp(1.0, max_ratio), nonterminal[i])
    kl_mask = 1.0 - (kl_div / kl_limit).clamp_max(1.0)
    c = p = (probs_ratio.clamp_max(max_ratio) * kl_mask).clamp_max(1.0)
    deltas = p * (rewards + nonterminal * discount * values[1:] - values[:-1])
    nonterm_c = nonterminal * c
    vs_minus_v_xs = torch.zeros_like(values)
    for i_inv in range(rewards.shape[0]):
        i = rewards.shape[0] - 1 - i_inv
        torch.addcmul(deltas[i], nonterm_c[i], vs_minus_v_xs[i + 1], value=discount, out=vs_minus_v_xs[i])
    value_targets = vs_minus_v_xs + values
    advantages_vtrace = rewards + nonterminal * discount * value_targets[1:] - values[:-1]

    assert value_targets.shape == values.shape, (value_targets.shape, values.shape)
    assert advantages_vtrace.shape == rewards.shape, (advantages_vtrace.shape, rewards.shape)

    return value_targets, advantages_vtrace, p


def assert_equal_tensors(a, b, abs_tol=1e-4):
    assert a.shape == b.shape, (a.shape, b.shape)
    assert (a - b).abs().max().item() < abs_tol, (a, b)


def test_upgo():
    rewards = torch.tensor([0, 0, 1], dtype=torch.float)
    dones = torch.tensor([0, 0, 0], dtype=torch.float)

    values_low = torch.tensor(     [0.5, 0.5, 0.5, 0], dtype=torch.float) # q = (0.5, 0.5, 1.0)
    upgo_target_low = torch.tensor([0.5, 1, 1], dtype=torch.float)
    upgo_low = calc_upgo(rewards.unsqueeze(1), values_low.unsqueeze(1), dones.unsqueeze(1), 1.0).squeeze(1)
    assert torch.allclose(upgo_low, upgo_target_low), upgo_low

    values_high = torch.tensor([1, 1, 2, 2], dtype=torch.float)
    upgo_high = calc_upgo(rewards.unsqueeze(1), values_high.unsqueeze(1), dones.unsqueeze(1), 1.0).squeeze(1)
    returns_high = calc_value_targets(rewards.unsqueeze(1), values_high.unsqueeze(1), dones.unsqueeze(1), 1.0).squeeze(1)
    assert torch.allclose(upgo_high, returns_high), upgo_high


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
    v_ret, v_adv, p = calc_vtrace(rewards, values, dones, prob_ratio, kl_div, discount, 2.0, 0.3)

    assert v_ret.shape == ret.shape
    assert v_adv.shape == adv.shape
    assert ((ret - v_ret).abs() > 1e-3).sum().item() == 0, (ret[-8:, 0], v_ret[-8:, 0])
    assert ((adv - v_adv).abs() > 1e-3).sum().item() == 0, (adv[-8:, 0], v_adv[-8:, 0])
