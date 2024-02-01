import math
from abc import abstractmethod, ABCMeta
from collections import OrderedDict
from functools import partial
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.init as init

from .heads import HeadBase, StateValueHead, PolicyHead, ActionValueHead
from .norm_factory import NormFactory
from .utils import weights_init
from ..common.attr_dict import AttrDict
from ..common.probability_distributions import make_pd, PointCloudPd


def create_ppo_actor(action_space, fx_factory, split_policy_value_network=True, num_values=1, is_recurrent=False):
    pd = make_pd(action_space)

    if split_policy_value_network:
        fx_policy, fx_value = fx_factory(), fx_factory()
    else:
        fx_policy = fx_value = fx_factory()

    value_head = StateValueHead(fx_value.output_size, num_out=num_values)
    policy_head = PolicyHead(fx_policy.output_size, pd=pd)
    if split_policy_value_network:
        models = OrderedDict([(fx_policy, dict(logits=policy_head)), (fx_value, dict(state_values=value_head))])
    else:
        models = OrderedDict([(fx_policy, dict(logits=policy_head, state_values=value_head))])
    return ModularActor(models, is_recurrent)


def create_impala_actor(action_space, fx_factory, split_policy_value_network, num_values, is_recurrent):
    pd = make_pd(action_space)

    if split_policy_value_network:
        fx_policy, fx_value = fx_factory(), fx_factory()
    else:
        fx_policy = fx_value = fx_factory()

    state_value_head = StateValueHead(fx_value.output_size, num_out=num_values)
    policy_head = PolicyHead(fx_policy.output_size, pd=pd, layer_norm=isinstance(pd, PointCloudPd))

    if split_policy_value_network:
        models = OrderedDict([
            (fx_policy, dict(logits=policy_head)),
            (fx_value, dict(state_values=state_value_head))])
    else:
        models = OrderedDict([
            (fx_policy, dict(logits=policy_head, state_values=state_value_head))])
    return ModularActor(models, is_recurrent)


def create_sac_actor(pd, policy_fx_factory, q_fx_factory, is_recurrent=False):
    fx_policy, fx_q1, fx_q2 = policy_fx_factory(), q_fx_factory(), q_fx_factory()

    q1_head = ActionValueHead(fx_q1.output_size, pd=pd)
    q2_head = ActionValueHead(fx_q2.output_size, pd=pd)
    policy_head = PolicyHead(fx_policy.output_size, pd=pd)
    models = {fx_policy: dict(logits=policy_head), fx_q1: dict(q1=q1_head), fx_q2: dict(q2=q2_head)}
    return ModularActor(models, is_recurrent)


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


def recursive_reset_weights(module):
    for m in module.children():
        if hasattr(m, 'reset_weights'):
            m.reset_weights()
        recursive_reset_weights(m)


class Actor(nn.Module, metaclass=ABCMeta):
    def reset_weights(self):
        recursive_reset_weights(self)

    @abstractmethod
    def forward(self, input, **kwargs) -> AttrDict:
        pass

    @property
    @abstractmethod
    def heads(self):
        pass


class ModularActor(Actor):
    def __init__(self, models: Dict[FeatureExtractorBase, Dict[str, HeadBase]], is_recurrent=False):
        super().__init__()
        self.models = models
        self.is_recurrent = is_recurrent
        self._heads = None
        self._fx_modules = None
        self._head_modules = None
        self._register_models()
        self.reset_weights()

    @property
    def heads(self):
        return self._heads

    @property
    def feature_extractors(self):
        return self._fx_modules

    def _register_models(self):
        self._heads = AttrDict({name: head for heads in self.models.values() for name, head in heads.items()})
        self._head_modules = nn.ModuleDict(self._heads)
        self._fx_modules = nn.ModuleList(self.models.keys())

    def forward(self, input, memory=None, evaluate_heads: List[str] = None, **kwargs) -> AttrDict:
        def need_head(name: str):
            return evaluate_heads is None or name in evaluate_heads

        fx_heads = [(fx, [(hn, h) for hn, h in heads.items() if need_head(hn)]) for (fx, heads) in self.models.items()]
        needed_fx = [fx if len(heads) > 0 else None for fx, heads in fx_heads]
        assert not self.is_recurrent or len(needed_fx) == len(fx_heads)
        output = self.run_fx(input, memory, needed_fx, **kwargs)

        for i, (fx, heads) in enumerate(fx_heads):
            for name, head in heads:
                if need_head(name):
                    output[name] = head(output[f'features_{i}'], **kwargs)

        return output

    def run_fx(self, input, memory=None, fx_list: List[Optional[FeatureExtractorBase]] = None, **kwargs):
        output = AttrDict()

        if fx_list is None:
            fx_list = self.feature_extractors
        assert len(fx_list) == len(self.feature_extractors)

        if memory is not None:
            memory_input = memory.chunk(len(self._fx_modules), dim=2)
        else:
            memory_input = [None] * len(self._fx_modules)

        memory_output = []

        for i, (fx, memory_input) in enumerate(zip(fx_list, memory_input)):
            if fx is None:
                continue

            if self.is_recurrent:
                features, memory = fx(input, memory=memory_input, **kwargs)
                memory_output.append(memory)
            else:
                features = fx(input, **kwargs)

            if isinstance(features, dict):
                for k, v in features.items():
                    output[f'{k}_{i}'] = v
            else:
                output[f'features_{i}'] = features

        if self.is_recurrent:
            output.memory = torch.cat(memory_output, dim=2)

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

    def head_fx_index(self, head_name):
        for i, (fx, heads) in enumerate(self.models.items()):
            if head_name in heads:
                return i
        raise KeyError

    def head_features(self, head_name, outputs):
        return outputs[f'features_{self.head_fx_index(head_name)}']
