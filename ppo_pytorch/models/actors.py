import math
from abc import ABC, abstractmethod, ABCMeta
from functools import partial
from typing import Optional, List, Callable, Dict, Tuple, Collection

import gym.spaces
import numpy as np
import torch.nn as nn
import torch.nn.init as init

from .heads import HeadBase
from .norm_factory import NormFactory
from .utils import weights_init
from ..common.probability_distributions import make_pd, ProbabilityDistribution
from ..common.attr_dict import AttrDict
import torch
import torch.nn.functional as F
import threading


class FeatureExtractorBase(nn.Module, metaclass=ABCMeta):
    def __init__(self, norm_factory: NormFactory=None,
                 weight_init_fn=partial(init.orthogonal_, gain=math.sqrt(2))):
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
    def forward(self, input, **kwargs):
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

    def forward(self, input, evaluate_heads: Collection[str]=None, **kwargs):
        output = AttrDict()
        for fx, heads in self.models.items():
            if evaluate_heads is not None and len(set(evaluate_heads) & set(heads.keys())) == 0:
                continue
            hidden = fx(input, **kwargs)
            for name, head in heads.items():
                if evaluate_heads is None or name in evaluate_heads:
                    output[name] = head(hidden, **kwargs)
        return output


# class Actor(nn.Module):
#     """
#     Base class for network in reinforcement learning algorithms.
#     """
#
#     def __init__(self, observation_space: gym.Space, action_space: gym.Space,
#                  head_factory: Callable[[int, ProbabilityDistribution], Dict[str, HeadBase]],
#                  norm: NormFactory = None, weight_init=init.orthogonal_, weight_init_gain=math.sqrt(2),
#                  hidden_code_type='input'):
#         """
#         Args:
#             observation_space: Observation space
#             action_space: Action space
#             head_factory: Function which accepts [hidden vector size, `ProbabilityDistribution`]
#                 and returns dict[head name, `HeadBase`]
#             norm: Normalization type
#             weight_init: Weight initialization function
#             weight_init_gain: Gain for `weight_init`
#         """
#         super().__init__()
#         self.observation_space = observation_space
#         self.action_space = action_space
#         self.head_factory = head_factory
#         self.weight_init_gain = weight_init_gain
#         self.weight_init = weight_init
#         self.norm = norm
#
#         self._thread_local = threading.local()
#         self.do_log = False
#         self.logger = None
#         self.hidden_code_size = None
#         self._step = 0
#         self.pd = make_pd(action_space)
#
#     def set_log(self, logger, do_log: bool, step: int):
#         """
#         Set logging state.
#         Args:
#             logger: Logger object.
#             do_log: Enable logging
#             step: Current training step
#         """
#         self.logger = logger
#         self.do_log = do_log
#         self._step = step
#
#     @property
#     def do_log(self):
#         return self._thread_local.do_log if hasattr(self._thread_local, 'do_log') else False
#
#     @do_log.setter
#     def do_log(self, value):
#         self._thread_local.do_log = value
#
#     def reset_weights(self):
#         self.apply(partial(weights_init, init_alg=self.weight_init, gain=self.weight_init_gain))
#         for m in self.modules():
#             if m is not self and hasattr(m, 'reset_weights'):
#                 m.reset_weights()
#         if hasattr(self, 'heads'):
#             for head in self.heads.values():
#                 head.reset_weights()
#
#     def _run_heads(self, hidden_code, heads=None):
#         heads = self.heads if heads is None else heads
#         heads = {name: head(hidden_code) for name, head in heads.items()}
#         return HeadOutput(hidden_code=hidden_code, **heads)
#
#     def _init_heads(self, hc_size):
#         self.heads = self._create_heads('heads', hc_size, self.pd, self.head_factory)
#
#     def _create_heads(self, head_name, hc_size, pd, head_factory):
#         heads = AttrDict(head_factory(hc_size, pd))
#         for name, head in heads.items():
#             self.add_module(f'{head_name}_{name}', head)
#         return heads
#
#     def __getstate__(self):
#         d = dict(self.__dict__)
#         d['logger'] = None
#         d['_thread_local'] = None
#         return d
#
#     def __setstate__(self, d):
#         self.__dict__ = d
#         self._thread_local = threading.local()
#
#
# class FCActor(Actor):
#     """
#     Fully connected network.
#     """
#
#     def __init__(self, observation_space: gym.Space, action_space: gym.Space, head_factory, *args,
#                  hidden_sizes=(128, 128), activation=nn.Tanh, hidden_code_type='input', **kwargs):
#         """
#         Args:
#             observation_space: Env's observation space
#             action_space: Env's action space
#             hidden_sizes: List of hidden layers sizes
#             activation: Activation function
#         """
#         super().__init__(observation_space, action_space, head_factory, *args, **kwargs)
#         self.hidden_sizes = hidden_sizes
#         self.activation = activation
#         self.hidden_code_type = hidden_code_type
#         obs_len = int(np.product(observation_space.shape))
#         self.hidden_code_size = obs_len if hidden_code_type == 'input' else hidden_sizes[-1]
#         self._init_heads(hidden_sizes[-1])
#         self.linear = self._create_fc(obs_len, None, hidden_sizes, activation, self.norm)
#         self.linear_value = self._create_fc(obs_len, None, hidden_sizes, activation, self.norm)
#         self.reset_weights()
#
#     def _run_linear(self, linear, input, hidden_code_input=False, only_hidden_code_output=False):
#         x = input
#         hidden_code = None
#         if self.hidden_code_type == 'last':
#             if hidden_code_input:
#                 hidden_code = input
#                 x = linear[-1][-1](input)
#             else:
#                 for i, layer in enumerate(linear):
#                     if i + 1 == len(linear):
#                         hidden_code = layer[:-1](x)
#                         if only_hidden_code_output:
#                             return HeadOutput(hidden_code=hidden_code)
#                         x = layer[-1](hidden_code)
#                     else:
#                         x = layer(x)
#                     if self.do_log:
#                         self.logger.add_histogram(f'layer {i} output', x, self._step)
#         else:
#             for i, layer in enumerate(linear):
#                 # x = layer(x) if self.input_as_hidden_code or i != 0 else layer[-1](x)
#                 if i == 0: # i + 1 == len(linear):
#                     hidden_code = x if hidden_code_input and self.hidden_code_type != 'input' else layer[:-1](x)
#                     if only_hidden_code_output:
#                         return HeadOutput(hidden_code=input if self.hidden_code_type == 'input' else hidden_code)
#                     x = layer[-1](hidden_code)
#                 else:
#                     x = layer(x)
#                 if self.do_log:
#                     self.logger.add_histogram(f'layer {i} output', x, self._step)
#         return x, hidden_code
#
#     def forward(self, input, hidden_code_input=False, only_hidden_code_output=False):
#         if hidden_code_input and only_hidden_code_output:
#             return HeadOutput(hidden_code=input)
#
#         x, hidden_code = self._run_linear(self.linear, input, hidden_code_input, only_hidden_code_output)
#         x_value, _ = self._run_linear(self.linear_value, input, hidden_code_input, only_hidden_code_output)
#
#         head = self._run_heads(x)
#         head.state_values = self._run_heads(x_value).state_values
#         if self.hidden_code_type == 'input':
#             hidden_code = input
#         head.hidden_code = hidden_code
#         return head
