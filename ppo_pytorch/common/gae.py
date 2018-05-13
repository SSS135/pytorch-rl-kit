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
    assert len(rewards) == len(values) - 1

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
    assert len(rewards) == len(values) - 1

    R = values[-1]
    returns = torch.zeros_like(rewards)
    for t in reversed(range(len(rewards))):
        nonterminal = 1 - dones[t]
        R = rewards[t] + nonterminal * reward_discount * R
        returns[t] = R

    return returns