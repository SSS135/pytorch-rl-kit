import math
from functools import partial
from typing import Optional, List, Callable, Dict

import gym.spaces
import numpy as np
import torch.nn as nn
import torch.nn.init as init

from .heads import HeadBase, HeadOutput
from .norm_factory import NormFactory
from .utils import weights_init
from ..common.probability_distributions import make_pd, ProbabilityDistribution
from ..common.attr_dict import AttrDict
import torch
import torch.nn.functional as F
import threading


class Actor(nn.Module):
    """
    Base class for network in reinforcement learning algorithms.
    """

    def __init__(self, observation_space: gym.Space, action_space: gym.Space,
                 head_factory: Callable[[int, ProbabilityDistribution], Dict[str, HeadBase]],
                 norm: NormFactory = None, weight_init=init.orthogonal_, weight_init_gain=math.sqrt(2),
                 hidden_code_type='input'):
        """
        Args:
            observation_space: Observation space
            action_space: Action space
            head_factory: Function which accepts [hidden vector size, `ProbabilityDistribution`]
                and returns dict[head name, `HeadBase`]
            norm: Normalization type
            weight_init: Weight initialization function
            weight_init_gain: Gain for `weight_init`
        """
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.head_factory = head_factory
        self.weight_init_gain = weight_init_gain
        self.weight_init = weight_init
        self.norm = norm

        self._thread_local = threading.local()
        self.do_log = False
        self.logger = None
        self.hidden_code_size = None
        self._step = 0
        self.pd = make_pd(action_space)

    def set_log(self, logger, do_log: bool, step: int):
        """
        Set logging state.
        Args:
            logger: Logger object.
            do_log: Enable logging
            step: Current training step
        """
        self.logger = logger
        self.do_log = do_log
        self._step = step

    @property
    def do_log(self):
        return self._thread_local.do_log if hasattr(self._thread_local, 'do_log') else False

    @do_log.setter
    def do_log(self, value):
        self._thread_local.do_log = value

    def reset_weights(self):
        self.apply(partial(weights_init, init_alg=self.weight_init, gain=self.weight_init_gain))
        for m in self.modules():
            if m is not self and hasattr(m, 'reset_weights'):
                m.reset_weights()
        if hasattr(self, 'heads'):
            for head in self.heads.values():
                head.reset_weights()

    @staticmethod
    def _create_fc(in_size: int, out_size: Optional[int], hidden_sizes: List[int],
                   activation: Callable, norm: NormFactory = None):
        """
        Create fully connected network
        Args:
            in_size: Input vector size.
            out_size: Optional. Output vector size. Additional layer is appended if not None.
            hidden_sizes: Width of hidden layers.
            activation: Activation function
            norm: Used normalization technique

        Returns: `nn.Sequential` of layers. Each layer is also `nn.Sequential` containing (linear, [norm], activation).
            If `out_size` is not None, last layer is just linear transformation, without norm or activation.

        """
        seq = []
        for i in range(len(hidden_sizes)):
            n_in = in_size if i == 0 else hidden_sizes[i - 1]
            n_out = hidden_sizes[i]
            layer = [nn.Linear(n_in, n_out, bias=norm is None or not norm.disable_bias)]
            if norm is not None and norm.allow_fc and (norm.allow_after_first_layer or i != 0):
                layer.append(norm.create_fc_norm(n_out, i == 0))
            layer.append(activation())
            seq.append(nn.Sequential(*layer))
        if out_size is not None:
            seq.append(nn.Linear(hidden_sizes[-1], out_size))
        seq = nn.Sequential(*seq)
        return seq

    def _run_heads(self, hidden_code, heads=None):
        heads = self.heads if heads is None else heads
        heads = {name: head(hidden_code) for name, head in heads.items()}
        return HeadOutput(hidden_code=hidden_code, **heads)

    def _init_heads(self, hc_size):
        self.heads = self._create_heads('heads', hc_size, self.pd, self.head_factory)

    def _create_heads(self, head_name, hc_size, pd, head_factory):
        heads = AttrDict(head_factory(hc_size, pd))
        for name, head in heads.items():
            self.add_module(f'{head_name}_{name}', head)
        return heads

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        del d['_thread_local']
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self._thread_local = threading.local()


class FCActor(Actor):
    """
    Fully connected network.
    """

    def __init__(self, observation_space: gym.Space, action_space: gym.Space, head_factory, *args,
                 hidden_sizes=(128, 128), activation=nn.Tanh, hidden_code_type='input', **kwargs):
        """
        Args:
            observation_space: Env's observation space
            action_space: Env's action space
            hidden_sizes: List of hidden layers sizes
            activation: Activation function
        """
        super().__init__(observation_space, action_space, head_factory, *args, **kwargs)
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.hidden_code_type = hidden_code_type
        obs_len = int(np.product(observation_space.shape))
        self.hidden_code_size = obs_len if hidden_code_type == 'input' else hidden_sizes[-1]
        self._init_heads(hidden_sizes[-1])
        self.linear = self._create_fc(obs_len, None, hidden_sizes, activation, self.norm)
        self.linear_value = self._create_fc(obs_len, None, hidden_sizes, activation, self.norm)
        self.reset_weights()

    def _run_linear(self, linear, input, hidden_code_input=False, only_hidden_code_output=False):
        x = input
        hidden_code = None
        if self.hidden_code_type == 'last':
            if hidden_code_input:
                hidden_code = input
                x = linear[-1][-1](input)
            else:
                for i, layer in enumerate(linear):
                    if i + 1 == len(linear):
                        hidden_code = layer[:-1](x)
                        if only_hidden_code_output:
                            return HeadOutput(hidden_code=hidden_code)
                        x = layer[-1](hidden_code)
                    else:
                        x = layer(x)
                    if self.do_log:
                        self.logger.add_histogram(f'layer {i} output', x, self._step)
        else:
            for i, layer in enumerate(linear):
                # x = layer(x) if self.input_as_hidden_code or i != 0 else layer[-1](x)
                if i == 0: # i + 1 == len(linear):
                    hidden_code = x if hidden_code_input and self.hidden_code_type != 'input' else layer[:-1](x)
                    if only_hidden_code_output:
                        return HeadOutput(hidden_code=input if self.hidden_code_type == 'input' else hidden_code)
                    x = layer[-1](hidden_code)
                else:
                    x = layer(x)
                if self.do_log:
                    self.logger.add_histogram(f'layer {i} output', x, self._step)
        return x, hidden_code

    def forward(self, input, hidden_code_input=False, only_hidden_code_output=False):
        if hidden_code_input and only_hidden_code_output:
            return HeadOutput(hidden_code=input)

        x, hidden_code = self._run_linear(self.linear, input, hidden_code_input, only_hidden_code_output)
        x_value, _ = self._run_linear(self.linear_value, input, hidden_code_input, only_hidden_code_output)

        head = self._run_heads(x)
        head.state_values = self._run_heads(x_value).state_values
        if self.hidden_code_type == 'input':
            hidden_code = input
        head.hidden_code = hidden_code
        return head
