import torch
import torch.nn as nn


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