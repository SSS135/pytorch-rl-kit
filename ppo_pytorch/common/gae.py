import math

import torch


def _check_data(rewards, values, dones):
    assert len(rewards) == len(dones) == len(values) - 1
    assert rewards.dim() == dones.dim() == 2
    assert values.dim() == 3


def calc_advantages(rewards, values, dones, reward_discount, advantage_discount):
    """
    Calculate advantages with Global Advantage Estimation
    Args:
        rewards: Rewards from environment
        values: State-values
        dones: Episode ended flags
        reward_discount: Discount factor for state-values
        advantage_discount: Discount factor for advantages

    Returns: GAE advantages
    """
    _check_data(rewards, values, dones)

    # adv_mult = (values[1:].std(-1) - values[:-1].std(-1)).abs()
    # # adv_mult = (values[1:] * reward_discount + rewards.unsqueeze(-1) - values[:-1]).std(-1)
    # adv_mult /= adv_mult.pow(2).mean().sqrt().add(0.01)

    values = values.mean(-1)

    next_adv = 0
    advantages = torch.zeros_like(rewards)
    for t in reversed(range(len(rewards))):
        nonterminal = 1 - dones[t]
        td_residual = rewards[t] + reward_discount * nonterminal * values[t + 1] - values[t]
        advantages[t] = next_adv = td_residual + advantage_discount * reward_discount * nonterminal * next_adv

    # advantages *= adv_mult

    return advantages


def calc_returns(rewards, values, dones, reward_discount):
    """
    Calculate temporal difference returns
    Args:
        rewards: Rewards from environment
        values: State-values
        dones: Episode ended flags
        reward_discount: Discount factor for state-values

    Returns: Temporal difference returns

    """
    _check_data(rewards, values, dones)
    rewards = rewards.unsqueeze(-1)
    dones = dones.unsqueeze(-1)

    R = values[-1]
    returns = rewards.new_zeros((*rewards.shape[:-1], values.shape[-1]))
    for t in reversed(range(len(rewards))):
        nonterminal = 1 - dones[t]
        R = rewards[t] + nonterminal * reward_discount * R
        returns[t] = R

    # vmean = values[:, 0].mean()
    # vstd = (values[1:, 0].std(-1) - values[:-1, 0].std(-1)).abs()
    # vstd2 = (values[1:, 0] * reward_discount + rewards[:, 0] - values[:-1, 0]).std(-1)
    # print(vmean, 'a', vstd, 'b', vstd2)

    return returns


def calc_vtrace(rewards, values, dones, probs_ratio, discount, c_max=2.0, p_max=2.0):
    c = probs_ratio.clamp(0, c_max)
    p = probs_ratio.clamp(0, p_max)
    nonterminal = 1 - dones
    td = p * (rewards + nonterminal * discount * values[1:] - values[:-1])
    targets = values.clone()
    for i in reversed(range(len(rewards))):
        targets[i] = values[i] + td[i] + nonterminal[i] * discount * c[i] * (targets[i + 1] - values[i + 1])
    advantages = rewards + nonterminal * discount * targets[1:] - values[:-1]
    return targets[:-1], advantages * p


def cumulative_value_mass_step(mass, decay):
    return math.log(1 + mass * math.log(decay)) / math.log(decay)


def cumulative_value_mass_fraction_step(fraction, decay):
    mass = fraction / (1 - decay)
    return math.log(1 + mass * math.log(decay)) / math.log(decay)


def chunk_value_mass(num_chunks, decay):
    torch.linspace()


def test():
    print()
    num_chunks = 10
    eps = 1 / (num_chunks)
    print(torch.linspace(eps, 1 - eps, num_chunks - 1))


def test_mass():
    assert math.isclose(cumulative_value_mass_step(50, 0.99), 69.4697, abs_tol=0.001)
    assert math.isclose(cumulative_value_mass_fraction_step(0.9, 0.99), 233.7181, abs_tol=0.001)


def test_vtrace():
    N = (1000, 8)
    discount = 0.99
    rewards = torch.randn(N)
    values = torch.randn((N[0] + 1, N[1]))
    dones = (torch.rand(N) > 0.95).float()
    cur_probs = old_probs = torch.zeros(N)
    ret = calc_returns(rewards, values, dones, discount)
    adv = calc_advantages(rewards, values, dones, discount, 1)
    v_ret, v_adv = calc_vtrace(rewards, values, dones, cur_probs / old_probs, discount, 1, 1)

    assert ((ret - v_ret).abs() > 1e-2).sum().item() == 0
    assert ((adv - v_adv).abs() > 1e-2).sum().item() == 0
