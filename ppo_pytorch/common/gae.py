import random

import math
from typing import Tuple, Optional

import torch
import torch.jit
from torch import Tensor


def _check_data(rewards: Tensor, values: Tensor, dones: Tensor):
    assert len(rewards) == len(dones) == len(values) - 1, (rewards.shape, dones.shape, values.shape)
    # (steps, actors)
    assert rewards.dim() == dones.dim() == 2, (rewards.shape, dones.shape, values.shape)
    # (steps, actors)
    assert values.dim() == 2, (rewards.shape, dones.shape, values.shape)


@torch.jit.script
def calc_advantages(rewards: Tensor, values: Tensor, dones: Tensor,
                    reward_discount: float, advantage_discount: float) -> Tensor:
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
def calc_advantages_noreward(rewards: Tensor, values: Tensor, dones: Tensor,
                             reward_discount: float, advantage_discount: float) -> Tensor:
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
def calc_value_targets(rewards: Tensor, values: Tensor, dones: Tensor,
                       reward_discount: float, gae_lambda: float = 1.0) -> Tensor:
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
def calc_upgo(rewards: Tensor, values: Tensor, dones: Tensor,
              reward_discount: float, lam: float = 1.0, action_values: Optional[Tensor] = None) -> Tensor:
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
    if action_values is None:
        action_values = rewards + nonterm_disc * values[1:]
    else:
        assert values.shape == action_values.shape
        action_values = action_values[:-1]
    target_factor = lam * (action_values > values[:-1]).float()

    for t_inv in range(rewards.shape[0]):
        t = rewards.shape[0] - 1 - t_inv
        torch.lerp(values[t], rewards[t] + nonterm_disc[t] * targets[t + 1], target_factor[t], out=targets[t])
        # lerp = target_factor[t + 1] if t_inv > 0 else target_factor[t]
        # targets[t] = rewards[t] + nonterm_disc[t] * torch.lerp(values[t + 1], targets[t + 1], lerp)

    return targets[:-1]


@torch.jit.script
def calc_vtrace(rewards: Tensor, values: Tensor, dones: Tensor, probs_ratio: Tensor,
                kl_div: Tensor, discount: float, kl_limit: float, lam: float) -> Tuple[Tensor, Tensor, Tensor]:
    _check_data(rewards, values, dones)
    assert probs_ratio.shape == rewards.shape == kl_div.shape == dones.shape, (probs_ratio.shape, rewards.shape, kl_div.shape)
    assert rewards.shape[0] == values.shape[0] - 1 and rewards.shape[1:] == values.shape[1:]

    nonterminal = 1 - dones
    kl_mask = (kl_div < kl_limit).float()
    # kl_mask = 1.0 - (kl_div / kl_limit).clamp_max(1.0)
    c = probs_ratio.clamp_max(lam) * kl_mask
    deltas = c * (rewards + nonterminal * discount * values[1:] - values[:-1])
    nonterm_c_disc = nonterminal * c * discount
    vs_minus_v_xs = torch.zeros_like(values)
    for i_inv in range(rewards.shape[0]):
        i = rewards.shape[0] - 1 - i_inv
        torch.addcmul(deltas[i], nonterm_c_disc[i], vs_minus_v_xs[i + 1], out=vs_minus_v_xs[i])
    value_targets = vs_minus_v_xs + values
    advantages = c * (rewards + nonterminal * discount * value_targets[1:] - values[:-1])

    assert value_targets.shape == values.shape, (value_targets.shape, values.shape)
    assert advantages.shape == rewards.shape, (advantages.shape, rewards.shape)

    return value_targets, advantages, c


@torch.jit.script
def calc_retrace(rewards: Tensor, state_values: Tensor, action_values: Tensor, dones: Tensor, probs_ratio: Tensor,
                 kl_div: Tensor, discount: float, kl_limit: float, lam: float) -> Tensor:
    _check_data(rewards, state_values, dones)
    assert probs_ratio.shape == rewards.shape == dones.shape, (probs_ratio.shape, rewards.shape)
    assert state_values.shape == action_values.shape
    assert rewards.shape[0] == state_values.shape[0] - 1 and rewards.shape[1:] == state_values.shape[1:]

    nonterminal = 1 - dones
    kl_mask = (kl_div < kl_limit).float()
    c = probs_ratio.clamp_max(lam) * kl_mask
    deltas = rewards + nonterminal * discount * state_values[1:] - action_values[:-1]
    nonterm_c_disc = nonterminal * c * discount
    vs_minus_v_xs = torch.zeros_like(state_values)
    for i_inv in range(rewards.shape[0]):
        i = rewards.shape[0] - 1 - i_inv
        torch.addcmul(deltas[i], nonterm_c_disc[i], vs_minus_v_xs[i + 1], out=vs_minus_v_xs[i])
    value_targets = vs_minus_v_xs + action_values

    assert value_targets.shape == state_values.shape, (value_targets.shape, state_values.shape)

    return value_targets


def assert_equal_tensors(a, b, abs_tol=1e-4):
    assert a.shape == b.shape, (a.shape, b.shape)
    assert (a - b).abs().max().item() < abs_tol, (a, b)


def test_upgo():
    rewards = Tensor([0, 0, 1], dtype=torch.float)
    dones = Tensor([0, 0, 0], dtype=torch.float)

    values_low = Tensor(     [0.5, 0.5, 0.5, 0], dtype=torch.float) # q = (0.5, 0.5, 1.0)
    upgo_target_low = Tensor([0.5, 1, 1], dtype=torch.float)
    upgo_low = calc_upgo(rewards.unsqueeze(1), values_low.unsqueeze(1), dones.unsqueeze(1), 1.0).squeeze(1)
    assert torch.allclose(upgo_low, upgo_target_low), upgo_low

    values_high = Tensor([1, 1, 2, 2], dtype=torch.float)
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
    v_ret, v_adv, p = calc_vtrace(rewards, values, dones, prob_ratio, kl_div, discount, 0.3, 1.0)
    v_ret = v_ret[:-1]

    assert v_ret.shape == ret.shape
    assert v_adv.shape == adv.shape
    assert ((ret - v_ret).abs() > 1e-3).sum().item() == 0, (ret[-8:, 0], v_ret[-8:, 0])
    assert ((adv - v_adv).abs() > 1e-3).sum().item() == 0, (adv[-8:, 0], v_adv[-8:, 0])
