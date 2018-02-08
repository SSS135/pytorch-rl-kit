# https://github.com/joschu/modular_rl/blob/master/modular_rl/core.py
# https://github.com/openai/baselines/blob/master/baselines/common/distributions.py

import math

import gym.spaces
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable


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
    def dtype_numpy(self):
        """Action's numpy data type"""
        raise NotImplementedError

    def dtype_torch(self, cuda):
        """Action's Pytorch data type"""
        raise NotImplementedError

    def neglogp(self, a, prob):
        """Negative log probability"""
        raise NotImplementedError

    def kl(self, prob0, prob1):
        """KL-Divergence"""
        raise NotImplementedError

    def entropy(self, prob):
        raise NotImplementedError

    @property
    def max_entropy(self):
        raise NotImplementedError

    def logp(self, a, prob):
        """Log probability"""
        return - self.neglogp(a, prob)

    def sample(self, prob):
        """Sample action from probabilities"""
        pass

    def to_inputs(self, action):
        """Convert actions to neural network input vector. For example, class number to one-hot vector."""
        return action


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
    def dtype_numpy(self):
        return np.int64

    def dtype_torch(self, cuda):
        return torch.cuda.LongTensor if cuda else torch.LongTensor

    def neglogp(self, a, prob):
        logp = log_softmax(prob)
        return -logp.gather(dim=-1, index=a.unsqueeze(-1) if a.dim() == 1 else a).squeeze(-1)

    def kl(self, prob0, prob1):
        logp0 = log_softmax(prob0)
        logp1 = log_softmax(prob1)
        return (logp0.exp() * (logp0 - logp1)).sum(dim=1)

    def entropy(self, prob):
        a = prob - prob.max(dim=-1, keepdim=True)[0]
        ea = torch.exp(a)
        z = ea.sum(dim=-1, keepdim=True)
        po = ea / z
        return torch.sum(po * (torch.log(z) - a), dim=-1)

    @property
    def max_entropy(self):
        return math.log(self.n)

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
    def dtype_numpy(self):
        return np.int64

    def dtype_torch(self, cuda):
        return torch.cuda.LongTensor if cuda else torch.LongTensor

    def neglogp(self, a, prob):
        return sigmoid_cross_entropy_with_logits(prob, a.float()).sum(-1)

    def kl(self, prob0, prob1):
        ps = sigmoid(prob0)
        return sigmoid_cross_entropy_with_logits(prob1, ps).sum(-1) - \
               sigmoid_cross_entropy_with_logits(prob0, ps).sum(-1)

    def entropy(self, prob):
        ps = sigmoid(prob)
        return sigmoid_cross_entropy_with_logits(prob, ps).sum(-1)

    @property
    def max_entropy(self):
        return self.n * math.log(2)  # TODO: check if correct

    def sample(self, prob):
        return sigmoid(prob).bernoulli()


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
    def dtype_numpy(self):
        return np.float32

    def dtype_torch(self, cuda):
        return torch.cuda.FloatTensor if cuda else torch.FloatTensor

    def neglogp(self, x, prob):
        mean = prob[:, :self.d]
        logstd = prob[:, self.d:]
        std = torch.exp(logstd)
        x = x.view_as(mean.data if isinstance(mean, Variable) else mean)
        nll = 0.5 * torch.sum(square((x - mean) / std), dim=-1) \
              + 0.5 * math.log(2.0 * math.pi) * self.d \
              + torch.sum(logstd, dim=-1)
        return nll.squeeze(-1)

    def kl(self, prob1, prob2):
        mean1 = prob1[:, :self.d]
        mean2 = prob2[:, :self.d]
        logstd1 = prob1[:, self.d:]
        logstd2 = prob2[:, self.d:]
        std1 = torch.exp(logstd1)
        std2 = torch.exp(logstd2)
        kl = torch.sum(logstd2 - logstd1 + (square(std1) + square(mean1 - mean2)) / (2.0 * square(std2)) - 0.5, dim=-1)
        return kl

    def entropy(self, prob):
        logstd = prob[:, self.d:]
        ent = torch.sum(logstd + .5 * math.log(2.0 * math.pi * math.e), dim=-1)
        return ent

    @property
    def max_entropy(self):
        return self.d * .5 * math.log(2.0 * math.pi * math.e)

    def sample(self, prob):
        mean = prob[:, :self.d]
        logstd = prob[:, self.d:]
        std = torch.exp(logstd)
        return mean + std * torch.randn(std.size()).type_as(std.data if isinstance(std, Variable) else std)


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