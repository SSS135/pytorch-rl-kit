import torch
import torch.nn.functional as F
from ppo_pytorch.common.squash import squash, unsquash
from torch import jit


class CategoricalValue:
    def __init__(self, num_bins, value_limit):
        super().__init__()
        self.num_bins = num_bins
        self.value_limit = value_limit
        vlim_squash = squash(torch.tensor(value_limit, dtype=torch.double)).item()
        self._support = unsquash(torch.linspace(-vlim_squash, vlim_squash, self.num_bins))

    def logits_to_value(self, logits):
        self._support = self._support.to(logits.device)
        with torch.no_grad():
            return (F.softmax(logits, -1) * self._support).sum(-1)

    def value_loss(self, logits, value_targets):
        self._support = self._support.to(logits.device)
        with torch.no_grad():
            min_bin, max_bin, bin_weight = self._cat_value_loss_bins(value_targets, self._support)
        return self._cat_value_loss_logits(logits, min_bin, max_bin, bin_weight)

    def project_dist(self, logits, rewards, gamma):
        assert rewards.shape == logits.shape[:-1]
        assert self._support.shape[-1] == logits.shape[-1]
        self._support = self._support.to(logits.device)
        with torch.no_grad():
            return self._project_dist(logits, rewards, gamma, self._support)

    @staticmethod
    @jit.script
    def _cat_value_loss_bins(value_targets, support):
        vdist = (support - value_targets.unsqueeze(-1)).abs()
        target_bin = vdist.min(-1)[1]
        target_bin_value = support[target_bin]
        close_bin = target_bin - (target_bin_value > value_targets).float().sub(0.5).sign().long()
        close_bin = close_bin.clamp(0, len(support) - 1)

        min_bin = torch.min(target_bin, close_bin)
        max_bin = torch.max(target_bin, close_bin)
        min_bin_value = support[min_bin]
        max_bin_value = support[max_bin]

        bin_weight = (value_targets - min_bin_value) / (max_bin_value - min_bin_value)
        one = torch.tensor(1.0, dtype=torch.float32, device=value_targets.device)
        bin_weight[min_bin == len(support) - 1] = -one
        bin_weight[max_bin == 0] = one
        return min_bin, max_bin, bin_weight

    @staticmethod
    @jit.script
    def _cat_value_loss_logits(logits, min_bin, max_bin, bin_weight):
        logp = F.log_softmax(logits, -1)
        logp_min = logp.gather(dim=-1, index=min_bin.unsqueeze(-1))
        logp_max = logp.gather(dim=-1, index=max_bin.unsqueeze(-1))
        bin_weight = bin_weight.unsqueeze(-1)
        return -(1 - bin_weight) * logp_min - bin_weight * logp_max

    @staticmethod
    # @jit.script
    def _project_dist(logits, rewards, gamma: float, support):
        proj_sup = gamma * support + rewards.unsqueeze(-1)

        vdist = (support.unsqueeze(-2) - proj_sup.unsqueeze(-1)).abs()
        target_bin = vdist.min(-1)[1]
        target_bin_value = support[target_bin]
        close_bin = target_bin - (target_bin_value > proj_sup).float().sub(0.5).sign().long()
        close_bin = close_bin.clamp(0, len(support) - 1)

        min_bin = torch.min(target_bin, close_bin)
        max_bin = torch.max(target_bin, close_bin)
        min_bin_value = support[min_bin]
        max_bin_value = support[max_bin]

        bin_weight = (proj_sup - min_bin_value) / (max_bin_value - min_bin_value)
        one = torch.tensor(1.0, dtype=torch.float32, device=proj_sup.device)
        bin_weight[min_bin == len(support) - 1] = -one
        bin_weight[max_bin == 0] = one

        # print('support', support)
        # print('proj_sup', proj_sup)
        # print('min_bin', min_bin)
        # print('max_bin', max_bin)
        # print('bin_weight', bin_weight)

        prob = F.softmax(logits, -1)
        prob_min = prob.gather(dim=-1, index=min_bin)
        prob_max = prob.gather(dim=-1, index=max_bin)
        proj_prob = torch.zeros_like(logits)
        proj_prob.scatter_add_(-1, min_bin, prob_min * (1 - bin_weight))
        proj_prob.scatter_add_(-1, max_bin, prob_max * bin_weight)

        # print('prob_min', prob_min)
        # print('prob_max', prob_max)
        # print('prob', prob)
        # print('proj_prob', proj_prob)

        return proj_prob


def test_dist():
    cat_value = CategoricalValue(3, 10)
    B = 500
    H = 32
    logits = torch.randn(B, cat_value.num_bins)
    rewards = torch.randn(H, B)
    gamma = 0.99
    # print()
    # print('logits', logits)
    # print('prob', F.softmax(logits, -1))
    # print('rewards', rewards)
    cat_value.project_dist(logits, rewards * gamma ** H, gamma)


def test_nondist():
    cat_value = CategoricalValue(51, 100)
    logits = torch.randn(1000, cat_value.num_bins, requires_grad=True)
    vtarg = torch.randn(1000) * 10
    optim = torch.optim.Adam([logits], lr=0.1)
    for i in range(1000):
        if (i + 1) % 100 == 0:
            print(i + 1, (cat_value.logits_to_value(logits) - vtarg).pow(2).mean().sqrt().item())
        cat_value.value_loss(logits, vtarg).sum().backward()
        optim.step()
        optim.zero_grad()
    assert (cat_value.logits_to_value(logits) - vtarg).pow(2).mean().sqrt().item() < 0.1
