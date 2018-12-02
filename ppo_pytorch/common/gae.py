import torch


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
    assert len(rewards) == len(values) - 1 == len(dones)

    gae = 0
    gaes = torch.zeros_like(rewards)
    for t in reversed(range(len(rewards))):
        nonterminal = 1 - dones[t]
        td_residual = rewards[t] + reward_discount * nonterminal * values[t + 1] - values[t]
        gaes[t] = gae = td_residual + advantage_discount * reward_discount * nonterminal * gae

    return gaes


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
    assert len(rewards) == len(values) - 1 == len(dones)

    R = values[-1]
    returns = torch.zeros_like(rewards)
    for t in reversed(range(len(rewards))):
        nonterminal = 1 - dones[t]
        R = rewards[t] + nonterminal * reward_discount * R
        returns[t] = R

    return returns


def calc_vtrace(rewards, values, dones, probs_ratio, discount, c_max=1, p_max=1):
    c = probs_ratio.clamp(0, c_max)
    p = probs_ratio.clamp(0, p_max)
    nonterminal = 1 - dones
    td = p * (rewards + nonterminal * discount * values[1:] - values[:-1])
    targets = values.clone()
    for i in reversed(range(len(rewards))):
        targets[i] = values[i] + td[i] + nonterminal[i] * discount * c[i] * (targets[i + 1] - values[i + 1])
    advantages = rewards + nonterminal * discount * targets[1:] - values[:-1]
    return targets[:-1], advantages, p


def test_vtrace():
    N = (1000, 8)
    discount = 0.99
    rewards = torch.randn(N)
    values = torch.randn((N[0] + 1, N[1]))
    dones = (torch.rand(N) > 0.95).float()
    cur_probs = old_probs = torch.zeros(N)
    ret = calc_returns(rewards, values, dones, discount)
    adv = calc_advantages(rewards, values, dones, discount, 1)
    v_ret, v_adv, _ = calc_vtrace(rewards, values, dones, cur_probs / old_probs, discount, 1, 1)

    assert ((ret - v_ret).abs() > 1e-2).sum().item() == 0
    assert ((adv - v_adv).abs() > 1e-2).sum().item() == 0
