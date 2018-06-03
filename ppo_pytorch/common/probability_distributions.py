# https://github.com/joschu/modular_rl/blob/master/modular_rl/core.py
# https://github.com/openai/baselines/blob/master/baselines/common/distributions.py

import math

import gym.spaces
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.distributions


def log_softmax(prob):
    """Same as `torch.nn.functional.log_softmax`, but accepts Tensors as well as Variables"""
    logp = F.log_softmax(prob if isinstance(prob, Variable) else Variable(prob), dim=-1)
    if not isinstance(prob, Variable):
        logp = logp.data
    return logp


def log_sigmoid(prob):
    """Same as `torch.nn.functional.logsigmoid`, but accepts Tensors as well as Variables"""
    logp = F.logsigmoid(prob if isinstance(prob, Variable) else Variable(prob))
    if not isinstance(prob, Variable):
        logp = logp.data
    return logp


def softmax(prob):
    """Same as `torch.nn.functional.softmax`, but accepts Tensors as well as Variables"""
    logp = F.softmax(prob if isinstance(prob, Variable) else Variable(prob), dim=-1)
    if not isinstance(prob, Variable):
        logp = logp.data
    return logp


def sigmoid(prob):
    """Same as `torch.nn.functional.sigmoid`, but accepts Tensors as well as Variables"""
    v = F.sigmoid(prob if isinstance(prob, Variable) else Variable(prob))
    if not isinstance(prob, Variable):
        v = v.data
    return v


def sigmoid_cross_entropy_with_logits(logits, labels):
    return torch.max(logits, 0)[0] - logits * labels + torch.log(1 + torch.exp(-logits.abs()))


def square(x):
    return x * x


def make_pd(space: gym.Space):
    """Create `ProbabilityDistribution` from gym.Space"""
    if isinstance(space, gym.spaces.Discrete):
        return CategoricalPd(space.n)
    elif isinstance(space, gym.spaces.Box):
        assert len(space.shape) == 1
        return DiagGaussianPd(space.shape[0])
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
        return -self.neglogp(a, prob)

    def neglogp(self, a, prob):
        """Negative log probability"""
        return -self.logp(a, prob)

    def sample(self, prob):
        """Sample action from probabilities"""
        pass

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

    def neglogp(self, a, prob):
        logp = log_softmax(prob)
        return -logp.gather(dim=-1, index=a.unsqueeze(-1) if a.dim() == 1 else a).squeeze(-1)

    def kl(self, prob0, prob1):
        logp0 = log_softmax(prob0)
        logp1 = log_softmax(prob1)
        return (logp0.exp() * (logp0 - logp1)).sum(dim=-1)

    def entropy(self, prob):
        a = prob - prob.max(dim=-1, keepdim=True)[0]
        ea = torch.exp(a)
        z = ea.sum(dim=-1, keepdim=True)
        po = ea / z
        return torch.sum(po * (torch.log(z) - a), dim=-1)

    def sample(self, prob):
        return softmax(prob).multinomial(1)

    def to_inputs(self, action):
        acvar = isinstance(action, Variable)
        onehot = torch.zeros((action.size(0), self.n))
        if action.is_cuda:
            onehot = onehot.cuda()
        onehot.scatter_(1, (action.data if acvar else action).view(action.size(0), -1), 1)
        if acvar:
            onehot = Variable(onehot)
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

    def neglogp(self, a, logits):
        nlp = F.binary_cross_entropy_with_logits(logits, a, reduce=False).sum(-1)
        return nlp

    def kl(self, prob0, prob1):
        ps = sigmoid(prob0)
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

    def neglogp(self, x, prob):
        mean = prob[..., :self.d]
        logstd = prob[..., self.d:]
        std = torch.exp(logstd)
        nll = 0.5 * ((x - mean) / std).pow(2) + \
              0.5 * math.log(2.0 * math.pi) * self.d + \
              logstd
        return nll.mean(-1)

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
        logstd = prob[..., self.d:]
        ent = logstd + .5 * math.log(2.0 * math.pi * math.e)
        return ent.mean(-1)

    def sample(self, prob):
        mean = prob[..., :self.d]
        logstd = prob[..., self.d:]
        std = torch.exp(logstd)
        return torch.normal(mean, std)

    @property
    def mean_div(self):
        return self.d


class FixedStdGaussianPd(ProbabilityDistribution):
    def __init__(self, d, std):
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

    def neglogp(self, x, prob):
        mean = prob[..., :self.d]
        var = self.std * self.std
        logvar = math.log(var)
        assert x.shape == mean.shape
        nll = 0.5 * ((x - mean) ** 2 / var + math.log(2 * math.pi) + logvar)
        return nll.sum(-1)

    def kl(self, prob1, prob2):
        mean1 = prob1[..., :self.d]
        mean2 = prob2[..., :self.d]
        logstd1 = math.log(self.std)
        logstd2 = math.log(self.std)
        std1 = math.exp(logstd1)
        std2 = math.exp(logstd2)
        kl = logstd2 - logstd1 + (square(std1) + square(mean1 - mean2)) / (2.0 * square(std2)) - 0.5
        return kl.sum(-1)

    def entropy(self, prob):
        logvar = prob.new(prob.shape[-1]).fill_(math.log(self.std * self.std))
        ent = 0.5 * (math.log(2 * math.pi * math.e) + logvar)
        return ent

    def sample(self, prob):
        mean = prob[..., :self.d]
        return torch.normal(mean, self.std)

    @property
    def mean_div(self):
        return self.d


class MultivecGaussianPd(ProbabilityDistribution):
    def __init__(self, d, num_vec, eps=1e-6):
        self.d = d
        self.num_vec = num_vec
        self.eps = eps

    @property
    def prob_vector_len(self):
        return self.d * self.num_vec

    @property
    def action_vector_len(self):
        return self.d

    @property
    def input_vector_len(self):
        return self.d

    @property
    def dtype(self):
        return torch.float

    def pdf(self, x, prob):
        vecs = prob.contiguous().view(*prob.shape[:-1], self.d, self.num_vec)
        var = vecs.var(-1).add(self.eps)
        mean = vecs.mean(-1)
        pdf = (2 * math.pi * var).rsqrt() + torch.exp(-(x - mean) ** 2 / (2 * var))
        return pdf

    def logp(self, x, prob):
        logp = self.pdf(x, prob).log()
        return logp

    def kl(self, prob1, prob2):
        vecs1 = prob1.contiguous().view(*prob1.shape[:-1], self.d, self.num_vec)
        vecs2 = prob2.contiguous().view(*prob2.shape[:-1], self.d, self.num_vec)
        var1 = vecs1.var(-1).add(self.eps)
        var2 = vecs2.var(-1).add(self.eps)
        mean1 = vecs1.mean(-1)
        mean2 = vecs2.mean(-1)
        std1 = var1.sqrt()
        std2 = var2.sqrt()
        logstd1 = std1.log()
        logstd2 = std2.log()
        kl = logstd2 - logstd1 + (square(std1) + square(mean1 - mean2)) / (2.0 * square(std2)) - 0.5
        return kl

    def entropy(self, prob):
        vecs = prob.contiguous().view(*prob.shape[:-1], self.d, self.num_vec)
        var = vecs.var(-1).add(self.eps)
        ent = 0.5 * (2 * math.pi * math.e * var).log()
        return ent

    def sample(self, prob):
        vecs = prob.contiguous().view(*prob.shape[:-1], self.d, self.num_vec)
        rand = torch.randint(self.num_vec, size=(vecs.shape[0],)).long()
        range = torch.arange(vecs.shape[0]).long()
        return vecs[range, ..., rand]

    @property
    def init_column_norm(self):
        return math.sqrt(self.d)


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
        return ent.mean(-1)

    def sample(self, mean, randn=None):
        if randn is None:
            randn = torch.randn_like(mean)
        return mean, randn

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

    def logp(self, a, prob):
        assert self.cur is not None and self.next is not None and self.randn is not None
        mean = prob[..., :self.d]
        std = prob[..., self.d:].exp()
        target = mean + std * self.randn
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

    def entropy(self, prob):
        logstd = prob[..., self.d:]
        mean = prob[..., :self.d]
        ent = logstd - mean ** 2 #+ .5 * math.log(2.0 * math.pi * math.e)
        return ent.mean(-1)

    def sample(self, prob, randn=None):
        mean = prob[..., :self.d]
        std = prob[..., self.d:].exp()
        if randn is None:
            randn = torch.randn_like(std)
        return mean + std * randn, randn

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