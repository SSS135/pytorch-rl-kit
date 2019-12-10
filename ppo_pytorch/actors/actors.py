import math
from abc import ABC, abstractmethod, ABCMeta
from functools import partial
from typing import Optional, List, Callable, Dict, Tuple, Collection, MutableMapping

import gym.spaces
import numpy as np
import torch.nn as nn
import torch.nn.init as init

from .heads import HeadBase, StateValueHead, PolicyHead
from .norm_factory import NormFactory
from .utils import weights_init
from ..common.probability_distributions import make_pd, ProbabilityDistribution, DiscretizedCategoricalPd
from ..common.attr_dict import AttrDict
import torch
import torch.nn.functional as F
import threading


def create_ppo_actor(action_space, fx_factory, split_policy_value_network=True, num_out=1):
    pd = make_pd(action_space)

    if split_policy_value_network:
        fx_policy, fx_value = fx_factory(), fx_factory()
    else:
        fx_policy = fx_value = fx_factory()

    value_head = StateValueHead(fx_value.output_size, pd=pd, num_out=num_out)
    policy_head = PolicyHead(fx_policy.output_size, pd=pd)
    if split_policy_value_network:
        models = {fx_policy: dict(logits=policy_head), fx_value: dict(state_values=value_head)}
    else:
        models = {fx_policy: dict(logits=policy_head, state_values=value_head)}
    return ModularActor(models)


def orthogonal_(tensor, gain=math.sqrt(2), mode='fan_in'):
    with torch.no_grad():
        tensor = torch.nn.init.orthogonal_(tensor)
        fan = init._calculate_correct_fan(tensor, mode)
        std = gain / math.sqrt(fan)
        rms = tensor.pow(2).mean().sqrt()
        return tensor.div_(rms).mul_(std)


class FeatureExtractorBase(nn.Module, metaclass=ABCMeta):
    def __init__(self, norm_factory: NormFactory=None,
                 weight_init_fn=orthogonal_):
        super().__init__()
        self.norm_factory = norm_factory
        self.weight_init_fn = weight_init_fn

    def reset_weights(self):
        self.apply(partial(weights_init, init_alg=self.weight_init_fn))

    @property
    @abstractmethod
    def output_size(self):
        pass


class Actor(nn.Module, metaclass=ABCMeta):
    def reset_weights(self):
        for m in self.modules():
            if m is not self and hasattr(m, 'reset_weights'):
                m.reset_weights()

    @abstractmethod
    def forward(self, input, **kwargs) -> AttrDict:
        pass

    @property
    @abstractmethod
    def heads(self):
        pass


class ModularActor(Actor):
    def __init__(self, models: Dict[FeatureExtractorBase, Dict[str, HeadBase]]):
        super().__init__()
        self.models = models
        self._heads = None
        self._fx_modules = None
        self._head_modules = None
        self._register_models()
        self.reset_weights()

    @property
    def heads(self):
        return self._heads

    def _register_models(self):
        self._heads = AttrDict({name: head for heads in self.models.values() for name, head in heads.items()})
        self._head_modules = nn.ModuleDict(self._heads)
        self._fx_modules = nn.ModuleList(self.models.keys())

    def forward(self, input, evaluate_heads: Collection[str]=None, **kwargs) -> AttrDict:
        output = AttrDict()
        for fx, heads in self.models.items():
            if evaluate_heads is not None and len(set(evaluate_heads) & set(heads.keys())) == 0:
                continue
            hidden = fx(input, **kwargs)
            for name, head in heads.items():
                if evaluate_heads is None or name in evaluate_heads:
                    output[name] = head(hidden, **kwargs)
        return output

    def head_parameters(self, *param_heads: str):
        match_found = False
        for fx, heads in self.models.items():
            matching_heads = [hval for hname, hval in heads.items() if hname in param_heads]
            if len(matching_heads) != 0:
                match_found = True
                for h in matching_heads:
                    yield from h.parameters()
                yield from fx.parameters()
        assert match_found