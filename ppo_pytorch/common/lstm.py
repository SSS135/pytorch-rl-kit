import torch
import torch.nn as nn


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
            hx, cx = mem.squeeze(0).chunk(2, -1)

        if reset_flags is not None:
            keep_flags = 1 - reset_flags

        outputs = []
        for step, x in enumerate(input):
            x = torch.cat([x, hx, cx], 1)
            net_out = self.net(x)
            hx, new_cx, gates = net_out.split([self.hidden_size, self.hidden_size, 3 * self.hidden_size], 1)
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


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)

    def forward(self, input, mem=None, reset_flags=None):
        # mem - (num_layers, batch_size, 2 * hidden_size)
        # input - (num_steps, batch_size, input_size)
        if mem is not None:
            mem = [x.contiguous() for x in mem.chunk(2, -1)]

        if reset_flags is not None:
            keep_flags = 1 - reset_flags

        outputs = []
        for step, x in enumerate(input):
            # x - (batch_size, input_size)
            # mem - tuple(2) of (num_layers, batch_size, hidden_size)
            # net_out - (num_steps, batch_size, hidden_size)
            net_out, mem = self.lstm(x.unsqueeze(0), mem)
            outputs.append(net_out.squeeze(0))

            if reset_flags is not None:
                keep = keep_flags[step].unsqueeze(-1)
                mem = (keep * mem[0], keep * mem[1])

        # (num_steps, batch_size, hidden_size)
        outputs = torch.stack(outputs, 0)
        return outputs, torch.cat(mem, -1)