# https://github.com/joschu/modular_rl/blob/master/modular_rl/core.py
# https://github.com/openai/baselines/blob/master/baselines/common/distributions.py

import math

import gym.spaces
import numpy as np
import torch
import torch.distributions
import torch.nn.functional as F


def make_pd(space: gym.Space):
    """Create `ProbabilityDistribution` from gym.Space"""
    if isinstance(space, gym.spaces.Discrete):
        return CategoricalPd(space.n)
    elif isinstance(space, gym.spaces.Box):
        assert len(space.shape) == 1
        return GaussianMixturePd(space.shape[0])
    elif isinstance(space, gym.spaces.MultiBinary):
        return BernoulliPd(space.n)
    else:
        raise TypeError(space)


class ProbabilityDistribution:
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

    def sample(self, prob):
        """Sample action from probabilities"""
        return self.sample_with_random(prob, None)[0]

    def sample_with_random(self, prob, rand):
        raise NotImplementedError()

    def to_inputs(self, action):
        """Convert actions to neural network input vector. For example, class number to one-hot vector."""
        return action

    @property
    def init_column_norm(self):
        return 0.01

    # @property
    # def mean_div(self):
    #     return 1
    #
    # @property
    # def clip_mult(self):
    #     return 1


class CategoricalPd(ProbabilityDistribution):
    def __init__(self, n):
        self.n = n

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

    def logp(self, a, prob):
        logp = F.log_softmax(prob, dim=-1)
        return logp.gather(dim=-1, index=a.unsqueeze(-1) if a.dim() == 1 else a).squeeze(-1)

    def kl(self, prob0, prob1):
        logp0 = F.log_softmax(prob0, dim=-1)
        logp1 = F.log_softmax(prob1, dim=-1)
        return (logp0.exp() * (logp0 - logp1)).sum(dim=-1)

    def entropy(self, prob):
        a = prob - prob.max(dim=-1, keepdim=True)[0]
        ea = a.exp()
        z = ea.sum(dim=-1, keepdim=True)
        po = ea / z
        return torch.sum(po * (torch.log(z) - a), dim=-1)

    def sample(self, prob):
        return F.softmax(prob, dim=-1).multinomial(1)

    def to_inputs(self, action):
        with torch.no_grad():
            onehot = torch.zeros((*action.shape[:-1], self.n), device=action.device)
            onehot.scatter_(dim=-1, index=action, value=1)
            onehot = onehot - 1 / self.n
        return onehot


class BernoulliPd(ProbabilityDistribution):
    def __init__(self, n):
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

    @property
    def mean_div(self):
        return self.n


class DiagGaussianPd(ProbabilityDistribution):
    def __init__(self, d):
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
        mean = prob[..., :self.d]
        logstd = prob[..., self.d:]
        std = torch.exp(logstd)
        nll = 0.5 * ((x - mean) / std).pow(2) + \
              0.5 * math.log(2.0 * math.pi) * self.d + \
              logstd
        return -nll

    def kl(self, prob1, prob2):
        mean1 = prob1[..., :self.d]
        mean2 = prob2[..., :self.d]
        logstd1 = prob1[..., self.d:]
        logstd2 = prob2[..., self.d:]
        std1 = torch.exp(logstd1)
        std2 = torch.exp(logstd2)
        kl = logstd2 - logstd1 + (std1 ** 2 + (mean1 - mean2) ** 2) / (2.0 * std2 ** 2) - 0.5
        return kl.mean(-1)

    def entropy(self, prob):
        mean = prob[..., :self.d]
        logstd = prob[..., self.d:]
        logvar = logstd * 2
        kld = logvar - logvar.exp() - mean.pow(2)
        return kld.mean(-1)
        # ent = logstd#.sign() * logstd.abs().add(1e-6).sqrt() #- mean ** 2 #+ .5 * math.log(2.0 * math.pi * math.e)
        # # ent[ent > 0] = 0
        # return ent.mean(-1)

    def sample_with_random(self, prob, rand):
        assert rand is None or torch.is_tensor(rand)
        mean = prob[..., :self.d]
        logstd = prob[..., self.d:]
        std = torch.exp(logstd)
        if rand is None:
            rand = torch.randn_like(mean)
        sample = mean + std * rand
        # sample = sample / sample.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
        return sample, rand

    @property
    def mean_div(self):
        return self.d


class GaussianMixturePd(ProbabilityDistribution):
    def __init__(self, d, num_mixtures=16, eps=1e-6):
        self.d = d
        self.num_mixtures = num_mixtures
        self.eps = eps
        self._gpd = DiagGaussianPd(d)
        self._cpd = CategoricalPd(num_mixtures)

    @property
    def prob_vector_len(self):
        return self.d * 2 * self.num_mixtures + self.num_mixtures

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
        logw, gaussians = self._split_prob(prob)
        logp = self._gpd.logp(x.unsqueeze(-2), gaussians) + F.log_softmax(logw, -1).unsqueeze(-1)
        return logp.mean(-1)

    def kl(self, prob1, prob2):
        logw1, gaussians1 = self._split_prob(prob1)
        logw2, gaussians2 = self._split_prob(prob2)
        kl = self._gpd.kl(gaussians1, gaussians2).mean(-1) + self._cpd.kl(logw1, logw2)
        return kl

    def entropy(self, prob):
        logw, gaussians = self._split_prob(prob)
        ent = self._gpd.entropy(gaussians).mean(-1) + self._cpd.entropy(logw)
        return ent

    def sample(self, prob):
        logw, gaussians = self._split_prob(prob)
        # (..., 1)
        mixture_idx = self._cpd.sample(logw)
        rep = *((gaussians.dim() - 1) * [1]), gaussians.shape[-1]
        index = mixture_idx.unsqueeze(-1).repeat(rep)
        selected = gaussians.gather(dim=-2, index=index).squeeze(-2)
        return self._gpd.sample(selected)

    # @property
    # def init_column_norm(self):
    #     return math.sqrt(self.d)

    def _split_prob(self, prob):
        logw, gaussians = prob.split([self.num_mixtures, self.d * 2 * self.num_mixtures], dim=-1)
        gaussians = gaussians.contiguous().view(*gaussians.shape[:-1], self.num_mixtures, self.d * 2)
        # logw - (..., n)
        # gaussians - (..., n, d * 2)
        return logw, gaussians


class StaticTransactionPd(ProbabilityDistribution):
    def __init__(self, d):
        self.d = d
        self.cur = None
        self.next = None
        self.randn = None

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
        assert self.cur is not None and self.next is not None and self.randn is not None
        target = mean
        real = self.next - self.cur
        p = F.cosine_similarity(real, target, dim=-1)
        return p

    def kl(self, mean1, mean2):
        def rmse(a, b):
            return (a - b).pow(2).mean(-1).add(1e-6).sqrt()
        return rmse(mean1, mean2) / 10

    def entropy(self, mean):
        ent = -mean ** 2
        ent = ent * 0
        return ent.mean(-1)

    def sample(self, mean, randn=None):
        if randn is None:
            randn = torch.randn_like(mean)
        sample = mean / mean.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
        return sample, randn

    @property
    def init_column_norm(self):
        return math.sqrt(self.d)


class DiagGaussianTransactionPd(ProbabilityDistribution):
    def __init__(self, d):
        self.d = d
        self.cur = None
        self.next = None
        self.randn = None

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
        assert self.cur is not None and self.next is not None and self.randn is not None
        target, _ = self.sample(prob, self.randn)
        # mean = prob[..., :self.d]
        # std = prob[..., self.d:].exp()
        # target = mean + std * self.randn
        # target = target / target.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
        real = self.next - self.cur
        p = F.cosine_similarity(real, target, dim=-1)
        # target = F.layer_norm(target, target.shape[-1:])
        # real = F.layer_norm(real, real.shape[-1:])
        # p = (target - real).pow(2).mean(-1).sqrt().neg()
        return p

    def kl(self, prob1, prob2):
        mean1 = prob1[..., :self.d]
        logstd1 = prob1[..., self.d:]
        mean2 = prob2[..., :self.d]
        logstd2 = prob2[..., self.d:]
        def rmse(a, b):
            return (a - b).pow(2).mean(-1).add(1e-6).sqrt()
        return (rmse(mean1, mean2) + rmse(logstd1, logstd2)) / 10

    # def entropy(self, prob):
    #     logstd = prob[..., self.d:]
    #     mean = prob[..., :self.d]
    #     ent = logstd #- mean.abs() #+ .5 * math.log(2.0 * math.pi * math.e)
    #     return ent.mean(-1)

    # def kl(self, prob1, prob2):
    #     mean1 = prob1[..., :self.d]
    #     mean2 = prob2[..., :self.d]
    #     logstd1 = prob1[..., self.d:]
    #     logstd2 = prob2[..., self.d:]
    #     std1 = torch.exp(logstd1)
    #     std2 = torch.exp(logstd2)
    #     kl = logstd2 - logstd1 + (std1 ** 2 + (mean1 - mean2) ** 2) / (2.0 * std2 ** 2) - 0.5
    #     return kl.mean(-1)

    def entropy(self, prob):
        mean = prob[..., :self.d]
        logstd = prob[..., self.d:]
        ent = logstd #- mean ** 2 #+ .5 * math.log(2.0 * math.pi * math.e)
        # ent[ent > 0] = 0
        return ent.mean(-1)

    def sample(self, prob, randn=None):
        mean = prob[..., :self.d]
        std = prob[..., self.d:].exp()
        if randn is None:
            randn = torch.randn_like(std)
        sample = mean + std * randn
        sample = sample / sample.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
        return sample, randn

    # @property
    # def mean_div(self):
    #     return self.d

    # @property
    # def clip_mult(self):
    #     return 0.1


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