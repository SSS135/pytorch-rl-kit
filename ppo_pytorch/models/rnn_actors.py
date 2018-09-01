import gym
import gym.spaces
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .actors import Actor


class DLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, activation=nn.Tanh):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.activation = activation

        layers = []
        for layer_idx in range(num_layers):
            last_layer = layer_idx == num_layers - 1
            in_features = hidden_size * 2 + input_size if layer_idx == 0 else hidden_size
            out_features = hidden_size * 5 if last_layer else hidden_size
            layers.append(nn.Linear(in_features, out_features))
            if not last_layer:
                layers.append(activation())
        self.net = nn.Sequential(*layers)
        self.hx_activation = self.activation()

    def forward(self, input, mem=None, reset_flags=None):
        # mem - (1, batch_size, 2 * hidden_size)
        # input - (num_steps, batch_size, input_size)
        if mem is None:
            cx = input.data.new(input.shape[1], self.hidden_size).zero_()
            hx = input.data.new(input.shape[1], self.hidden_size).zero_()
        else:
            cx, hx = mem.squeeze(0).chunk(2, -1)

        if reset_flags is not None:
            keep_flags = 1 - reset_flags

        outputs = []
        for step, x in enumerate(input):
            x = torch.cat([x, cx, hx], 1)
            net_out = self.net(x)
            new_cx, hx, gates = net_out.split([self.hidden_size, self.hidden_size, 3 * self.hidden_size], 1)
            o_gate, f_gate, i_gate = gates.sigmoid().chunk(3, 1)

            if reset_flags is not None:
                keep = keep_flags[step].unsqueeze(-1)
                i_gate *= keep
                f_gate *= 1 - keep

            hx = self.hx_activation(hx) * o_gate
            new_cx = self.hx_activation(new_cx)
            cx = i_gate * new_cx + f_gate * cx
            outputs.append(hx)
        # (num_steps, batch_size, hidden_size)
        outputs = torch.stack(outputs, 0)
        return outputs, torch.cat([hx, cx], -1).unsqueeze(0)


class DLSTMActor(Actor):
    def __init__(self, observation_space: gym.Space, action_space: gym.Space, *args,
                 hidden_code_size=128, num_layers=2, **kwargs):
        """
        Args:
            observation_space: Env's observation space
            action_space: Env's action space
            head_factory: Function which accept (hidden vector size, `ProbabilityDistribution`) and return `HeadBase`
            hidden_code_size: Hidden layer width
            activation: Activation function
        """
        super().__init__(observation_space, action_space, *args, **kwargs)
        self.num_layers = num_layers
        self.hidden_code_size = hidden_code_size
        obs_len = int(np.product(observation_space.shape))
        self.rnn = DLSTM(obs_len, self.hidden_code_size, self.num_layers)
        self._init_heads(self.hidden_code_size)
        self.reset_weights()

    def forward(self, input, memory, done_flags):
        x, next_memory = self.rnn(input, memory, done_flags)
        head = self._run_heads(x)
        head.hidden_code = x
        return head, next_memory