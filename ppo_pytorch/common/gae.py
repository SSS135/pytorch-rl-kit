import math

import torch


def _check_data(rewards, values, dones):
    assert len(rewards) == len(dones) == len(values) - 1
    # (steps, actors)
    assert rewards.dim() == dones.dim() == 2
    # (steps, actors, bins, q)
    assert values.dim() == 4


def calc_advantages(rewards, values, dones, reward_discount, advantage_discount):
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

    # adv_mult = (values[1:].std(-1) - values[:-1].std(-1)).abs()
    # # adv_mult = (values[1:] * reward_discount + rewards.unsqueeze(-1) - values[:-1]).std(-1)
    # adv_mult /= adv_mult.pow(2).mean().sqrt().add(0.01)

    values = values.mean(-1).sum(-1)

    next_adv = 0
    advantages = torch.zeros_like(rewards)
    for t in reversed(range(len(rewards))):
        nonterminal = 1 - dones[t]
        td_residual = rewards[t] + reward_discount * nonterminal * values[t + 1] - values[t]
        advantages[t] = next_adv = td_residual + advantage_discount * reward_discount * nonterminal * next_adv

    # advantages *= adv_mult

    return advantages


def calc_weighted_advantages(rewards, values, dones, reward_discount, advantage_discount):
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

    # adv_mult = (values[1:].std(-1) - values[:-1].std(-1)).abs()
    adv_mult = (values[1:] * reward_discount + rewards.unsqueeze(-1) - values[:-1])
    adv_mult = adv_mult.abs().mean(-1)
    adv_mult /= adv_mult.pow(2).mean().sqrt().add(0.01)

    # values = values.mean(-1).sum(-1)

    next_adv = 0
    advantages = torch.zeros_like(rewards)
    for t in reversed(range(len(rewards))):
        nonterminal = 1 - dones[t]
        td_residual = rewards[t] + reward_discount * nonterminal * values[t + 1] - values[t]
        advantages[t] = next_adv = td_residual + advantage_discount * reward_discount * nonterminal * next_adv

    advantages *= adv_mult

    return advantages


def calc_value_targets(rewards, values, dones, reward_discount, gae_lambda=1.0):
    """
    Calculate temporal difference targets
    Args:
        rewards: Rewards from environment (steps, actors)
        values: State-values (steps, actors, bins, q)
        dones: Episode ended flags (steps, actors)
        reward_discount: Discount factor for state-values

    Returns: Target values (steps, actors, bins, q)
    """
    _check_data(rewards, values, dones)

    rewards = rewards.unsqueeze(-1).unsqueeze(-1)
    dones = dones.unsqueeze(-1).unsqueeze(-1)

    R = values[-1]
    targets = rewards.new_zeros((values.shape[0] - 1, *values.shape[1:]))
    for t in reversed(range(len(rewards))):
        nonterminal = 1 - dones[t]
        R = (1 - gae_lambda) * values[t] + gae_lambda * (rewards[t] + nonterminal * reward_discount * R)
        targets[t] = R

    return targets


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


def calc_binned_value_targets(rewards, values, dones, reward_discount):
    # (steps, actors, bins, q)
    assert values.dim() == 4
    # (steps, actors)
    assert rewards.dim() == dones.dim() == 2
    num_steps, num_actors, num_bins, num_q = values.shape
    assert rewards.shape == dones.shape == (num_steps - 1, num_actors)

    # (num_unbinned_rewards, num_actors, num_q)
    per_step_rewards = split_binned_values(values[-1], reward_discount)
    # (len(rewards) + num_unbinned_rewards, num_actors, num_q)
    per_step_rewards = torch.cat([rewards.unsqueeze(-1).expand(*rewards.shape, num_q), per_step_rewards], 0)
    targets = torch.zeros_like(values[:-1])
    for i in reversed(range(len(targets))):
        per_step_rewards[i + 1:] *= reward_discount * (1 - dones[i].unsqueeze(-1))
        targets[i] = values_to_bins(per_step_rewards[i:], num_bins, reward_discount)
    return targets


def values_to_bins(values, num_bins, reward_discount):
    # (num_steps, num_actors, num_q) -> (num_actors, num_bins, num_q)
    assert values.dim() == 3
    num_steps, num_actors, num_q = values.shape
    pivots = get_value_pivots(num_bins, reward_discount).tolist()
    assert len(pivots) == num_bins - 1
    bins = values.new_zeros((num_actors, num_bins, num_q))
    for i, pivot in enumerate(pivots):
        start = 0 if i == 0 else pivots[i - 1]
        bins[:, i] = values[start: pivot].sum(0)
    bins[:, -1] = values[0 if len(pivots) == 0 else pivots[-1]:].sum(0)
    return bins


def split_binned_values(values, reward_discount):
    # (num_actors, num_bins, num_q) -> (num_steps, num_actors, num_q)
    assert values.dim() == 3
    num_actors, num_bins, num_q = values.shape
    pivots = get_value_pivots(num_bins, reward_discount).tolist()
    assert len(pivots) == num_bins - 1
    step_rewards = []
    for i, pivot in enumerate(pivots):
        start = 0 if i == 0 else pivots[i - 1]
        count = pivot - start
        cur = values[:, i].div(count).unsqueeze(0).expand(count, num_actors, num_q)
        lambdas = reward_discount ** torch.arange(count, device=cur.device, dtype=cur.dtype)
        lambdas /= lambdas.mean()
        step_rewards.append(cur * lambdas.view(-1, 1, 1))
    step_rewards.append(values[:, -1].unsqueeze(0))
    step_rewards = torch.cat(step_rewards, 0)
    assert step_rewards.shape == ((pivots[-1] + 1) if len(pivots) != 0 else 1, num_actors, num_q)
    return step_rewards


def get_value_pivots(num_bins, decay):
    assert num_bins >= 1
    assert 1 >= decay >= 0
    if num_bins == 1:
        return torch.LongTensor([])
    eps = 1 / num_bins
    fraction = torch.linspace(eps, 1 - eps, num_bins - 1)
    mass = fraction / (1 - decay)
    pivots = torch.log(1 + mass * math.log(decay)) / math.log(decay)
    return torch.round(pivots).long()


def assert_equal_tensors(a, b, abs_tol=1e-4):
    assert a.shape == b.shape, (a.shape, b.shape)
    assert (a - b).abs().max().item() < abs_tol, (a, b)


def test_binned_value_targets_single_bin():
    rewards = torch.tensor([1, 2, 3, 5, 10]).float().view(-1, 1)
    values_bin = torch.tensor([4, 8, 12, 16, 20, 24]).float().view(-1, 1, 1, 1)
    dones = torch.tensor([0, 0, 0, 0, 0]).float().view(-1, 1)
    discount = 0.9

    req_targets = calc_value_targets(rewards, values_bin, dones, discount).unsqueeze(-2)
    out_targets = calc_binned_value_targets(rewards, values_bin, dones, discount)
    assert_equal_tensors(req_targets, out_targets)


def test_binned_value_targets_rand():
    num_steps = 128
    num_actors = 64
    num_bins = 8
    num_q = 16

    rewards = torch.randn((num_steps, num_actors))
    values_bin = torch.randn((num_steps + 1, num_actors, num_bins, num_q))
    dones = torch.rand((num_steps, num_actors)).lt_(0.01)
    discount = 0.99

    req_targets = calc_value_targets(rewards, values_bin, dones, discount).unsqueeze(-2)
    out_targets = calc_binned_value_targets(rewards, values_bin, dones, discount).sum(-2, keepdim=True)
    assert_equal_tensors(req_targets, out_targets)


def test_binned_value_targets_multi_bin():
    rewards = torch.tensor([1, 2, 3]).float().view(-1, 1)
    values_bin = torch.tensor([[1, 1, 1], [1, 2, 3], [1, 2, 3], [4, 3, 2]]).float().view(-1, 1, 3, 1)
    dones = torch.tensor([0, 0, 0]).float().view(-1, 1)
    discount = 0.9

    out_targets = calc_binned_value_targets(rewards, values_bin, dones, discount)
    print(out_targets)


def test_get_value_pivots():
    pivots_target = torch.tensor([11, 22, 36, 51, 69, 92, 121, 162, 234])
    pivots_result = get_value_pivots(10, 0.99)
    assert_equal_tensors(pivots_target, pivots_result)


def test_split_binned_values():
    split_target = torch.tensor([0.2500, 0.2500, 0.2500, 0.2500, 0.6250, 0.6250, 0.6250,
                                 0.6250, 0.6250, 0.6250, 0.6250, 0.6250, 3.0000]).view(-1, 1, 1)
    binned_steps = torch.tensor([1.0, 5, 3]).view(1, -1, 1)
    discount = 0.9

    split_result = split_binned_values(binned_steps, discount)
    assert_equal_tensors(split_target, split_result)

    rebinned_steps = values_to_bins(split_result, binned_steps.shape[1], 0.9)
    assert_equal_tensors(rebinned_steps, binned_steps)


def test_split_binned_values_rand():
    num_actors = 64
    num_bins = 8
    num_q = 16
    discount = 0.99

    values_bin = torch.rand((num_actors, num_bins, num_q))

    split_result = split_binned_values(values_bin, discount)
    rebinned_steps = values_to_bins(split_result, num_bins, discount)

    assert_equal_tensors(rebinned_steps, values_bin)


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
