# https://github.com/joschu/modular_rl/blob/master/modular_rl/core.py
# https://github.com/openai/baselines/blob/master/baselines/common/distributions.py

import math
from typing import Dict, Any, List

import gymnasium as gym
import numpy as np
import torch
import torch.distributions
import torch.jit
import torch.nn.functional as F
from torch.distributions import Beta, kl_divergence, Normal, Gumbel


def make_pd(space: gym.Space):
    """Create `ProbabilityDistribution` from gym.Space"""
    if isinstance(space, gym.spaces.Discrete):
        return CategoricalPd(space.n)
        # return GumbelCategoricalPd(space.n)
    elif isinstance(space, gym.spaces.Box):
        assert len(space.shape) == 1
        # return LinearTanhPd(space.shape[0])
        # return FixedStdGaussianPd(space.shape[0], 0.3)
        # return BetaPd(space.shape[0], 1)
        # return DiagGaussianPd(space.shape[0])
        # return MixturePd(space.shape[0], 4, partial(BetaPd, h=1))
        # return PointCloudPd(space.shape[0], max_norm=2.0, cloud_size=16)
        return DiscretizedCategoricalPd(space.shape[0], 9, ordinal=True)
    elif isinstance(space, gym.spaces.MultiBinary):
        return BernoulliPd(space.n)
    elif isinstance(space, gym.spaces.MultiDiscrete):
        return MultiCategoricalPd(space.nvec)
    elif isinstance(space, gym.spaces.Dict):
        return MultiPd([make_pd(s) for s in space.spaces.values()])
    else:
        raise TypeError(space)


class ProbabilityDistribution:
    def __init__(self, init_args: Dict[str, Any]):
        del init_args['self']
        del init_args['__class__']
        self._init_args = init_args

    """Unified API to work with different types of probability distributions"""
    @property
    def prob_vector_len(self):
        """Length of policy output vector,
        i.e. number of actions for categorical and gaussian"""
        raise NotImplementedError

    @property
    def action_vector_len(self):
        """Length of action vector which passed to `env.step()`,
        i.e. one for categorical, number of actions for gaussian"""
        raise NotImplementedError

    @property
    def input_vector_len(self):
        """Length of action vector if used as input to neural network,
        i.e. one-hot vector length"""
        raise NotImplementedError

    @property
    def dtype(self):
        """Action data type"""
        raise NotImplementedError

    def kl(self, prob0, prob1):
        """KL-Divergence"""
        raise NotImplementedError

    def entropy(self, prob):
        raise NotImplementedError

    def logp(self, a, prob):
        """Log probability"""
        raise NotImplementedError

    def oob_loss(self, prob) -> float:
        """Out of bounds loss"""
        return 0

    def sample(self, prob):
        """Sample action from probabilities"""
        raise NotImplementedError()

    def rsample(self, prob):
        """Sample with gradient"""
        raise NotImplementedError()

    def to_inputs(self, action):
        """Convert actions to neural network input vector. For example, class number to one-hot vector."""
        return action

    def postprocess_action(self, action):
        """Action postprocessing such as tanh or clamping"""
        return action

    def __repr__(self):
        str_args = str.join(', ', [f'{k}={v}' for k, v in self._init_args.items()])
        return f'{self.__class__.__name__}({str_args})'

    __str__ = __repr__


class CategoricalPd(ProbabilityDistribution):
    def __init__(self, n, ordinal=False):
        super().__init__(locals())
        self.n = n
        self.ordinal = ordinal

    @property
    def prob_vector_len(self):
        return self.n

    @property
    def action_vector_len(self):
        return 1

    @property
    def input_vector_len(self):
        return self.n

    @property
    def dtype(self):
        return torch.int64

    def logp(self, a, logits):
        logits = self._process_logits(logits)
        lsm = F.log_softmax(logits, dim=-1)
        logp = lsm.gather(dim=-1, index=a)
        assert logp.shape == (*logits.shape[:-1], 1)
        return logp

    def kl(self, logits0, logits1):
        logits0 = self._process_logits(logits0)
        logits1 = self._process_logits(logits1)
        # return (logits0 - logits1).pow(2).mean(-1, keepdim=True)
        logp0 = F.log_softmax(logits0, dim=-1)
        logp1 = F.log_softmax(logits1, dim=-1)
        return (logp0.exp() * (logp0 - logp1)).sum(dim=-1, keepdim=True)

    def entropy(self, logits):
        logits = self._process_logits(logits)
        logits = logits - logits.logsumexp(dim=-1, keepdim=True)
        p_log_p = logits * logits.softmax(-1)
        return -p_log_p.sum(-1, keepdim=True)

    def sample(self, logits):
        with torch.no_grad():
            logits = self._process_logits(logits)
            return F.softmax(logits.reshape(-1, logits.shape[-1]), dim=-1)\
                .multinomial(1).reshape(*logits.shape[:-1], -1)

    def to_inputs(self, action):
        with torch.no_grad():
            onehot = torch.zeros((*action.shape[:-1], self.n), device=action.device)
            onehot.scatter_(dim=-1, index=action, value=1)
            return onehot

    def _process_logits(self, logits):
        assert logits.shape[-1] == self.n, (logits.shape[-1], self.n)
        return make_logits_ordnial(logits) if self.ordinal else logits


class GumbelCategoricalPd(CategoricalPd):
    def __init__(self, n, ordinal=False, temp=1.0):
        super().__init__(n, ordinal)
        self.temp = temp
        self.gumbel = Gumbel(0, 1)

    def sample(self, logits):
        with torch.no_grad():
            logits = self._process_logits(logits)
            logits = logits + self.gumbel.sample(logits.shape).to(logits.device)
            return logits.max(-1)[1].unsqueeze(-1)

    def rsample(self, logits):
        logits = self._process_logits(logits)
        logits = (logits + self.gumbel.sample(logits.shape).to(logits.device)) / self.temp
        return F.softmax(logits, dim=-1)

    def to_inputs(self, action):
        if action.is_floating_point():
            assert action.shape[-1] == self.n
            return action
        return super().to_inputs(action)


class MultiPd(ProbabilityDistribution):
    def __init__(self, pds: List[ProbabilityDistribution]):
        super().__init__(locals())
        self.pds = pds
        self._prob_sizes = [pd.prob_vector_len for pd in pds]
        self._actions_sizes = [pd.action_vector_len for pd in pds]
        self._input_sizes = [pd.input_vector_len for pd in pds]
        self._dtype = pds[0].dtype if all(p.dtype == pds[0].dtype for p in pds) else torch.float32

    @property
    def prob_vector_len(self):
        return sum(self._prob_sizes)

    @property
    def action_vector_len(self):
        return sum(self._actions_sizes)

    @property
    def input_vector_len(self):
        return sum(self._input_sizes)

    @property
    def dtype(self):
        return self._dtype

    def logp(self, all_actions, all_logits):
        split_logits = all_logits.split(self._prob_sizes, -1)
        split_actions = all_actions.split(self._actions_sizes, -1)
        all_logp = [pd.logp(a.type(pd.dtype), logits) for pd, logits, a in zip(self.pds, split_logits, split_actions)]
        return torch.cat(all_logp, -1)

    def kl(self, all_logits0, all_logits1):
        split_logits0 = all_logits0.split(self._prob_sizes, -1)
        split_logits1 = all_logits1.split(self._prob_sizes, -1)
        all_kl = [pd.kl(logits0, logits1) for pd, logits0, logits1 in zip(self.pds, split_logits0, split_logits1)]
        return torch.cat(all_kl, -1)

    def entropy(self, all_logits):
        split_logits = all_logits.split(self._prob_sizes, -1)
        all_ent = [pd.entropy(logits) for pd, logits in zip(self.pds, split_logits)]
        return torch.cat(all_ent, -1)

    def sample(self, all_logits):
        with torch.no_grad():
            split_logits = all_logits.split(self._prob_sizes, -1)
            all_actions = [pd.sample(logits).type(self.dtype) for pd, logits in zip(self.pds, split_logits)]
            return torch.cat(all_actions, -1)

    def rsample(self, all_logits):
        split_logits = all_logits.split(self._prob_sizes, -1)
        all_actions = [pd.rsample(logits).type(self.dtype) for pd, logits in zip(self.pds, split_logits)]
        return torch.cat(all_actions, -1)

    def to_inputs(self, all_actions):
        with torch.no_grad():
            split_actions = all_actions.split(self._actions_sizes, -1)
            all_inputs = [pd.to_inputs(action.type(pd.dtype)).type(torch.float32) for pd, action in zip(self.pds, split_actions)]
            return torch.cat(all_inputs, -1)


class MultiCategoricalPd(MultiPd):
    def __init__(self, sizes: List[int]):
        super().__init__([CategoricalPd(s) for s in sizes])


class BernoulliPd(ProbabilityDistribution):
    def __init__(self, n):
        super().__init__(locals())
        self.n = n

    @property
    def prob_vector_len(self):
        return self.n

    @property
    def action_vector_len(self):
        return self.n

    @property
    def input_vector_len(self):
        return self.n

    @property
    def dtype(self):
        return torch.int64

    def logp(self, a, logits):
        logp = -F.binary_cross_entropy_with_logits(logits, a, reduce=False).sum(-1)
        return logp

    def kl(self, prob0, prob1):
        ps = F.sigmoid(prob0)
        kl = F.binary_cross_entropy_with_logits(prob1, ps, reduce=False).sum(-1) - \
             F.binary_cross_entropy_with_logits(prob0, ps, reduce=False).sum(-1)
        return kl

    def entropy(self, logits):
        probs = logits.sigmoid()
        ent = F.binary_cross_entropy_with_logits(logits, probs, reduce=False).sum(-1)
        return ent

    def sample(self, prob):
        with torch.no_grad():
            return prob.sigmoid().bernoulli()


class DiagGaussianPd(ProbabilityDistribution):
    LOG_STD_MAX = 2
    LOG_STD_MIN = -6

    def __init__(self, d):
        super().__init__(locals())
        self.d = d

    @property
    def prob_vector_len(self):
        return self.d * 2

    @property
    def action_vector_len(self):
        return self.d

    @property
    def input_vector_len(self):
        return self.d

    @property
    def dtype(self):
        return torch.float

    def logp(self, x, prob):
        mean, logstd = self._split_probs(prob)
        std = logstd.exp()
        logp = -0.5 * (
            ((x - mean) / (std + 1e-6)) ** 2
            + 2 * logstd
            + np.log(2 * np.pi)
        )
        return logp

    def kl(self, prob1, prob2):
        mean1, logstd1 = self._split_probs(prob1)
        mean2, logstd2 = self._split_probs(prob2)
        std1, std2 = logstd1.exp(), logstd2.exp()
        dist1 = Normal(mean1, std1)
        dist2 = Normal(mean2, std2)
        return kl_divergence(dist1, dist2)

    def entropy(self, prob):
        mean, logstd = self._split_probs(prob)
        return 0.5 * (
            math.log(2 * np.pi * np.e) + 2 * logstd
        )

    def oob_loss(self, prob):
        mean, logstd = self._split_probs(prob)
        return mean.abs().clamp_min_(1).mean() + \
               (logstd > self.LOG_STD_MAX).float().mul_(logstd).mean() + \
               -(logstd < self.LOG_STD_MIN).float().mul_(logstd).mean()

    def sample(self, prob):
        with torch.no_grad():
            return self.rsample(prob)

    def rsample(self, prob):
        mean, logstd = self._split_probs(prob)
        std = logstd.exp()
        return mean + std * torch.randn_like(mean)

    def _split_probs(self, probs):
        mean, logstd = probs.chunk(2, -1)
        return mean, logstd.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)

    def postprocess_action(self, action):
        return action.clamp(-1, 1)


class PointCloudPd(ProbabilityDistribution):
    def __init__(self, d, cloud_size=32, apply_tanh=True, max_norm=1.0):
        super().__init__(locals())
        self.d = d
        self.cloud_size = cloud_size
        self.apply_tanh = apply_tanh
        self.max_norm = max_norm

    @property
    def prob_vector_len(self):
        return self.d * self.cloud_size

    @property
    def action_vector_len(self):
        return self.d

    @property
    def input_vector_len(self):
        return self.d

    @property
    def dtype(self):
        return torch.float

    def logp(self, x, prob):
        mean, std = self._get_mean_std(prob)
        logstd = std.log()
        logp = -0.5 * (
            ((x - mean) / (std + 1e-6)) ** 2
            + 2 * logstd
            + np.log(2 * np.pi)
        )
        if self.apply_tanh:
            logp = logp - 2 * (math.log(2) - x - F.softplus(-2 * x))
        return logp

    def kl(self, prob1, prob2):
        mean1, std1 = self._get_mean_std(prob1)
        mean2, std2 = self._get_mean_std(prob2)
        dist1 = Normal(mean1, std1)
        dist2 = Normal(mean2, std2)
        return kl_divergence(dist1, dist2)

    def entropy(self, prob):
        mean, std = self._get_mean_std(prob)
        return 0.5 * (
            math.log(2 * np.pi * np.e) + 2 * std.log()
        )

    def sample(self, prob):
        with torch.no_grad():
            return self.rsample(prob)

    def rsample(self, prob):
        prob = self._split_probs(prob)
        idx = torch.randint(0, self.cloud_size, prob.shape[:-2], device=prob.device)
        idx = torch.zeros((*prob.shape[:-1], 1), dtype=idx.dtype, device=prob.device) + idx.unsqueeze(-1).unsqueeze(-1)
        action = prob.gather(-1, idx).squeeze(-1)
        return action

    def _split_probs(self, probs):
        # probs = probs / probs.pow(2).mean(-1, keepdim=True).add(1e-7).sqrt()
        probs = probs.reshape(*probs.shape[:-1], self.d, self.cloud_size)
        probs = limit_abs_mean(probs, self.max_norm, dim=-2)
        return probs

    def _get_mean_std(self, prob):
        prob = self._split_probs(prob)
        mean = prob.mean(-1)
        std = (prob.var(-1) + 1e-8).sqrt()
        return mean, std

    def postprocess_action(self, action):
        return action.tanh() if self.apply_tanh else action


class BetaPd(ProbabilityDistribution):
    def __init__(self, d, h, eps=1e-7):
        super().__init__(locals())
        self.d = d
        self.h = h
        self.eps = eps

    @property
    def prob_vector_len(self):
        return self.d * 2

    @property
    def action_vector_len(self):
        return self.d

    @property
    def input_vector_len(self):
        return self.d

    @property
    def dtype(self):
        return torch.float

    def logp(self, x, prob):
        p = (x * (1 - 1e-5) + self.h) / (2 * self.h)

        mask = (p < self.eps) | (p > 1 - self.eps)
        assert mask.sum() == 0, x[mask]

        beta = self._beta(prob)
        logp = beta.log_prob(p)
        return logp

    def kl(self, prob1, prob2):
        beta1 = self._beta(prob1)
        beta2 = self._beta(prob2)
        kl = kl_divergence(beta1, beta2)
        return kl

    def entropy(self, prob):
        beta = self._beta(prob)
        ent = beta.entropy()
        return ent

    def sample(self, prob):
        beta = self._beta(prob)
        sample = 2 * self.h * (beta.sample() - 0.5)
        return sample

    def _beta(self, prob):
        prob = 1 + F.softplus(prob)
        return Beta(*prob.chunk(2, -1))


class FixedStdGaussianPd(ProbabilityDistribution):
    def __init__(self, d, std):
        super().__init__(locals())
        self.d = d
        self.std = std

    @property
    def prob_vector_len(self):
        return self.d

    @property
    def action_vector_len(self):
        return self.d

    @property
    def input_vector_len(self):
        return self.d

    @property
    def dtype(self):
        return torch.float

    def logp(self, x, mean):
        assert x.shape == mean.shape
        return -((x - mean) ** 2) / 2 - math.log(math.sqrt(2 * math.pi))

    def kl(self, mean1, mean2):
        dist1 = Normal(mean1, self.std)
        dist2 = Normal(mean2, self.std)
        return kl_divergence(dist1, dist2)

    def entropy(self, mean):
        return torch.zeros_like(mean) + (0.5 + 0.5 * math.log(2 * math.pi) + math.log(self.std))

    def oob_loss(self, mean):
        return mean.abs().clamp_min_(1).mean()

    def sample(self, mean):
        with torch.no_grad():
            return self.rsample(mean)

    def rsample(self, mean):
        return mean + self.std * torch.randn_like(mean)

    def postprocess_action(self, action):
        return action.clamp(-1, 1)


@torch.compile
def limit_abs_mean(v: torch.Tensor, limit: float, dim: int = -1):
    len = v.abs().mean(dim, keepdim=True).clamp(min=limit)
    return v * (limit / len)


@torch.compile
def limit_abs_max(v: torch.Tensor, limit: float, dim: int = -1):
    len = v.abs().max(dim, keepdim=True)[0].clamp(min=limit)
    return v * (limit / len)


class DiscretizedCategoricalPd(ProbabilityDistribution):
    def __init__(self, d, num_bins, ordinal=True):
        super().__init__(locals())
        self.d = d
        self.num_bins = num_bins
        self.cpd = MultiPd([CategoricalPd(num_bins, ordinal) for _ in range(d)])

    @property
    def prob_vector_len(self):
        return self.cpd.prob_vector_len

    @property
    def action_vector_len(self):
        return self.d

    @property
    def input_vector_len(self):
        return self.d

    @property
    def dtype(self):
        return torch.float

    def logp(self, action, logits):
        bin_indexes = (action + 1) * ((self.num_bins - 1.0) / 2)
        bin_indexes = bin_indexes.round().long()
        return self.cpd.logp(bin_indexes, logits)

    def kl(self, logits1, logits2):
        return self.cpd.kl(logits1, logits2)

    def entropy(self, logits):
        return self.cpd.entropy(logits)

    def sample(self, logits):
        bin_indexes = self.cpd.sample(logits)
        return bin_indexes.float() * (2 / (self.num_bins - 1.0)) - 1



@torch.compile
def make_logits_ordnial(logits):
    logits_pos = F.logsigmoid(logits).unsqueeze(-2)
    logits_neg = F.logsigmoid(-logits).unsqueeze(-2)
    upper_tri = torch.ones(logits.shape[-1], logits.shape[-1], device=logits.device).triu(1)
    next_sum = (upper_tri * logits_neg).sum(-1)
    prev_cur_sum = ((1 - upper_tri) * logits_pos).sum(-1)
    return prev_cur_sum + next_sum


def test_probtypes():
    np.random.seed(0)

    pdparam_categorical = np.array([-.2, .3, .5])
    categorical = CategoricalPd(pdparam_categorical.size) #pylint: disable=E1101
    validate_probtype(categorical, pdparam_categorical)

    pdparam_diag_gauss = np.array([-.2, .3, .4, -.5, .1, -.5, .1, 0.8])
    diag_gauss = DiagGaussianPd(pdparam_diag_gauss.size // 2) #pylint: disable=E1101
    validate_probtype(diag_gauss, pdparam_diag_gauss)

    pdparam_bernoulli = np.array([-.2, .3, .5])
    bernoulli = BernoulliPd(pdparam_bernoulli.size) #pylint: disable=E1101
    validate_probtype(bernoulli, pdparam_bernoulli)


def validate_probtype(pd, pdparam):
    N = 100000
    M = np.repeat(pdparam[None, :], N, axis=0)
    M = torch.from_numpy(M).float()
    X = pd.sample(M)

    # Check to see if mean negative log likelihood == differential entropy
    logliks = pd.logp(X, M)
    entval_ll = - logliks.mean()
    entval_ll_stderr = logliks.std() / np.sqrt(N)
    entval = pd.entropy(M).mean()
    assert np.abs(entval - entval_ll) < 3 * entval_ll_stderr # within 3 sigmas

    # Check to see if kldiv[p,q] = - ent[p] - E_p[log q]
    M2 = M + torch.randn(M.shape) * 0.1
    klval = pd.kl(M, M2).mean()
    logliks = pd.logp(X, M2)
    klval_ll = - entval - logliks.mean() #pylint: disable=E1101
    klval_ll_stderr = logliks.std() / np.sqrt(N) #pylint: disable=E1101
    assert np.abs(klval - klval_ll) < 3 * klval_ll_stderr # within 3 sigmas