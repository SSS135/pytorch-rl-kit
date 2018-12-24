import math
from typing import Tuple

import torch
from torch import Tensor as TT


def _check_data(rewards: TT, values: TT, dones: TT):
    assert len(rewards) == len(dones) == len(values) - 1, (rewards.shape, dones.shape, values.shape)
    # (steps, actors)
    assert rewards.dim() == dones.dim() == 2, (rewards.shape, dones.shape, values.shape)
    # (steps, actors, bins, q)
    assert values.dim() == 4, (rewards.shape, dones.shape, values.shape)


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

    values = values.mean(-1).sum(-1)

    next_adv = 0
    advantages = torch.zeros_like(rewards)
    for t in reversed(range(len(rewards))):
        nonterminal = 1 - dones[t]
        td_residual = rewards[t] + reward_discount * nonterminal * values[t + 1] - values[t]
        advantages[t] = next_adv = td_residual + advantage_discount * reward_discount * nonterminal * next_adv

    return advantages


def calc_value_targets(rewards: TT, values: TT, dones: TT, reward_discount: float, gae_lambda=1.0) -> TT:
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
    assert values.shape[-2] == 1

    rewards = rewards.unsqueeze(-1).unsqueeze(-1)
    dones = dones.unsqueeze(-1).unsqueeze(-1)

    R = values[-1]
    targets = rewards.new_zeros((values.shape[0] - 1, *values.shape[1:]))
    for t in reversed(range(len(rewards))):
        nonterminal = 1 - dones[t]
        R = rewards[t] + nonterminal * reward_discount * R
        targets[t] = R
        R = (1 - gae_lambda) * values[t] + gae_lambda * R

    return targets


def calc_weighted_advantages(rewards: TT, values: TT, dones: TT, reward_discount: float, advantage_discount: float) -> TT:
    _check_data(rewards, values, dones)

    # (steps, actors, bins, q)
    assert values.dim() == 4
    # (steps, actors)
    assert rewards.dim() == dones.dim() == 2
    values = values.mean(-1, keepdim=True)
    _, num_actors, num_bins, _ = values.shape
    num_steps = values.shape[0] - 1
    assert rewards.shape == dones.shape == (num_steps, num_actors)

    # (steps + 1, unbinned_rewards, actors, q)
    rewards_from_values = split_binned_values(values, reward_discount).transpose_(1, 2).contiguous()
    num_unb_rewards = rewards_from_values.shape[1]
    td_errors = values.new_zeros((num_steps, num_steps + num_unb_rewards, num_actors, 1))

    # (num_steps + num_unb_rewards, num_actors, 1)
    per_step_rewards = torch.cat([rewards.unsqueeze(-1).expand(*rewards.shape, 1), rewards_from_values[-1]], 0)

    def get_value_cur(step):
        # (steps + unb_rewards, actors, q)
        v_targ = torch.cat([
            # (i, actors, 1) or empty
            *((rewards.new_zeros((step, 1, 1)).expand(-1, num_actors, 1),) if step != 0 else ()),
            # (rfv_last_split_steps + unbinned_rewards, actors, q)
            resize_last_reward(
                rewards_from_values[step], td_errors.shape[1] - step, reward_discount)
        ], 0)
        assert v_targ.shape == td_errors.shape[1:]
        return v_targ

    def get_value_target(step):
        # (steps + unb_rewards, actors, q)
        v_targ = torch.cat([
            # (i, actors, 1) or empty
            *((rewards.new_zeros((step, 1, 1)).expand(-1, num_actors, 1),) if step != 0 else ()),
            per_step_rewards[step:],
        ], 0)
        assert v_targ.shape == td_errors.shape[1:]
        return v_targ

    # calc td errors
    for i in reversed(range(num_steps)):
        per_step_rewards[i + 1:] *= reward_discount * (1 - dones[i].unsqueeze(-1))

        value_cur = get_value_cur(i)
        value_target = get_value_target(i)
        td_errors[i] = value_target - value_cur

        per_step_rewards[i:] *= advantage_discount
        per_step_rewards[i:] += (1 - advantage_discount) * resize_last_reward(
                rewards_from_values[i], td_errors.shape[1] - i, reward_discount)

    ep_start = [0] * num_actors
    dones_list = dones.cpu().tolist()
    for step in range(0, num_steps):
        for ac_i in range(num_actors):
            if dones_list[step][ac_i]:
                slc = (slice(ep_start[ac_i], step + 1), None, ac_i)
                td = td_errors[slc]
                td_sum = td.abs().sum(0, keepdim=True)
                td /= (td_sum + 1e-6)
                td *= td_sum.sum(1, keepdim=True) / (td.abs().sum(0, keepdim=True).sum(1, keepdim=True) + 1e-6)
                td_errors[slc] = td
                ep_start[ac_i] = step + 1

    advantages = td_errors.sum(1).mean(-1)

    return advantages


def resize_last_reward(rewards: TT, desired_size: int, discount: float) -> TT:
    # (steps, actors, q) -> (desired_size, actors, q)
    cur_size = rewards.shape[0]
    assert desired_size >= cur_size
    total_mass = 1 / (1 - discount)
    cur_last_mass_left = total_mass - get_mass(cur_size - 1, discount)
    desired_last_mass_left = total_mass - get_mass(desired_size - 1, discount)
    desired_last_value = rewards[-1] * (desired_last_mass_left / cur_last_mass_left)
    expanded_values_sum = rewards[-1] - desired_last_value
    resized_rewards = rewards.new_zeros((desired_size, *rewards.shape[1:]))
    resized_rewards[:rewards.shape[0] - 1] = rewards[:-1]
    if desired_size > cur_size:
        resized_rewards[rewards.shape[0] - 1: -1] = \
            expanded_values_sum.unsqueeze(0) / (desired_size - cur_size) * \
            new_discount_weights(rewards, desired_size - cur_size, discount).view(-1, 1, 1)
    resized_rewards[-1] = desired_last_value
    return resized_rewards


def calc_binned_value_targets(rewards: TT, values: TT, dones: TT, reward_discount: float, advantage_discount: float=1.0) -> TT:
    # (steps, actors, bins, q)
    assert values.dim() == 4
    # (steps, actors)
    assert rewards.dim() == dones.dim() == 2
    num_steps, num_actors, num_bins, num_q = values.shape
    assert rewards.shape == dones.shape == (num_steps - 1, num_actors)

    # (steps + 1, unbinned_rewards, actors, q)
    all_per_step_rewards = split_binned_values(values, reward_discount).transpose(1, 2).contiguous()
    # (len(rewards) + num_unbinned_rewards, num_actors, num_q)
    per_step_rewards = torch.cat([rewards.unsqueeze(-1).expand(*rewards.shape, num_q), all_per_step_rewards[-1]], 0)
    targets = torch.zeros_like(values[:-1])
    for i in reversed(range(len(targets))):
        per_step_rewards[i + 1:] *= reward_discount * (1 - dones[i].unsqueeze(-1))
        targets[i] = values_to_bins(per_step_rewards[i:], num_bins, reward_discount)
        per_step_rewards[i:] *= advantage_discount
        per_step_rewards[i:i + all_per_step_rewards.shape[1]] += (1 - advantage_discount) * all_per_step_rewards[i]
    return targets


def values_to_bins(values: TT, num_bins: int, reward_discount: float) -> TT:
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


def new_discount_weights(self: TT, len: int, discount: float) -> TT:
    assert len > 0
    lambdas = discount ** torch.arange(len, device=self.device, dtype=self.dtype)
    lambdas /= lambdas.mean()
    return lambdas


def split_binned_values(values: TT, reward_discount: float) -> TT:
    # (num_values, num_actors, num_bins, num_q) -> (num_values, num_actors, num_steps, num_q)
    assert values.dim() == 4
    num_values, num_actors, num_bins, num_q = values.shape
    pivots = get_value_pivots(num_bins, reward_discount).tolist()
    assert len(pivots) == num_bins - 1
    step_rewards = []
    for i, pivot in enumerate(pivots):
        start = 0 if i == 0 else pivots[i - 1]
        count = pivot - start
        cur = values[:, :, i].div(count).unsqueeze(2).expand(num_values, num_actors, count, num_q)
        lambdas = new_discount_weights(cur, count, reward_discount).view(1, 1, -1, 1)
        step_rewards.append(cur * lambdas)
    step_rewards.append(values[:, :, -1].unsqueeze(2))
    step_rewards = torch.cat(step_rewards, 2)
    assert step_rewards.shape == (num_values, num_actors, (pivots[-1] + 1) if len(pivots) != 0 else 1, num_q)
    return step_rewards


def get_value_pivots(num_bins: int, decay: float) -> TT:
    assert num_bins >= 1
    assert 1 >= decay >= 0
    if num_bins == 1:
        return torch.LongTensor([])
    eps = 1 / num_bins
    fraction = torch.linspace(eps, 1 - eps, num_bins - 1)
    mass = fraction / (1 - decay)
    pivots = get_step(mass, decay)
    return torch.round(pivots).long()


def get_step(mass, decay):
    return torch.log((decay - 1) * mass + 1) / math.log(decay) - 1


def get_mass(step, decay):
    return (decay ** (step + 1) - 1) / (decay - 1)


def calc_vtrace(rewards: TT, values: TT, dones: TT, probs_ratio: TT, discount: float, c_max=1.5, p_max=1.5) -> Tuple[TT, TT]:
    _check_data(rewards, values, dones)
    assert values.shape[-2] == 1, values.shape
    assert probs_ratio.shape == rewards.shape, (probs_ratio.shape, rewards.shape)

    rewards = rewards.unsqueeze(-1).unsqueeze(-1)
    dones = dones.unsqueeze(-1).unsqueeze(-1)
    probs_ratio = probs_ratio.unsqueeze(-1).unsqueeze(-1)

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

    advantages = (advantages * p).mean(-1).mean(-1)

    return targets[:-1], advantages


def assert_equal_tensors(a, b, abs_tol=1e-4):
    assert a.shape == b.shape, (a.shape, b.shape)
    assert (a - b).abs().max().item() < abs_tol, (a, b)


def test_mass_step():
    lam = 0.99
    mass = torch.tensor([1.0, 15, 50, 90])
    step = get_step(mass, lam)
    print('\n', step)
    assert_equal_tensors(mass, get_mass(step, lam))


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
