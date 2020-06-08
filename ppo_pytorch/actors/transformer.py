import torch
from ppo_pytorch.common.silu import SiLU, silu
from torch import nn


class GRUCellLast(nn.GRUCell):
    def forward(self, input, hx):
        return super().forward(input.view(-1, input.shape[-1]), hx.view(-1, hx.shape[-1])).view_as(hx)


class SimpleTrLayer(nn.Module):
    def __init__(self, hidden_size, emb_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.num_heads = num_heads

        self.attn = nn.MultiheadAttention(num_heads * emb_size, num_heads)
        self.seq_kv_linear = nn.Linear(hidden_size, 2 * num_heads * emb_size)
        self.main_q_linear = nn.Linear(hidden_size, num_heads * emb_size)
        self.layer_norm_main = nn.LayerNorm(hidden_size)
        self.layer_norm_seq = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            SiLU(),
        )
        self.gru = nn.ModuleList([nn.GRUCell(hidden_size, hidden_size), nn.GRUCell(hidden_size, hidden_size)])

    def reset_weights(self):
        with torch.no_grad():
            for gru in self.gru:
                gru.reset_parameters()
                gru.bias_hh.fill_(0)
                gru.bias_ih.fill_(0)
                gru.bias_hh[self.hidden_size:2 * self.hidden_size] += 1

    def forward(self, main, seq):
        main = self.gru[0](self._attention(main, seq), main)
        main = self.gru[1](self.mlp(main), main)
        return main

    def _attention(self, main, seq):
        B, NU, H = seq.shape
        assert main.shape == (B, H)
        q = self.main_q_linear(self.layer_norm_main(main))\
            .view(1, B, self.num_heads * self.emb_size)
        k, v = self.seq_kv_linear(self.layer_norm_seq(seq))\
            .view(B, NU, 2, self.num_heads * self.emb_size).transpose(0, 1).unbind(-2)
        attn = self.attn(q, k, v)[0].squeeze(0)
        assert attn.shape == main.shape, (attn.shape, main.shape)
        return silu(attn)


class TrLayer(nn.Module):
    def __init__(self, hidden_size, emb_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.num_heads = num_heads

        self.attn = nn.MultiheadAttention(num_heads * emb_size, num_heads)
        self.seq_qkv_linear = nn.Linear(hidden_size, 3 * num_heads * emb_size)
        self.layer_norm_seq = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            SiLU(),
        )

    def forward(self, seq):
        seq = seq + self._attention(seq)
        seq = seq + self.mlp(seq)
        return seq

    def _attention(self, seq):
        B, NU, H = seq.shape
        q, k, v = self.seq_qkv_linear(self.layer_norm_seq(seq))\
            .view(B, NU, 3, self.num_heads * self.emb_size).transpose(0, 1).unbind(-2)
        attn = self.attn(q, k, v)[0].transpose(0, 1)
        assert attn.shape == seq.shape, (attn.shape, seq.shape)
        return silu(attn)


class TrPriorFirstLayer(nn.Module):
    def __init__(self, hidden_size, emb_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.num_heads = num_heads

        self.attn = nn.MultiheadAttention(num_heads * emb_size, num_heads)
        self.seq_qkv_linear = nn.Linear(hidden_size, 3 * num_heads * emb_size)
        self.main_qkv_linear = nn.Linear(hidden_size, 3 * num_heads * emb_size)
        self.layer_norm_seq = nn.LayerNorm(hidden_size)
        self.layer_norm_main = nn.LayerNorm(hidden_size)
        self.mlp_main = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            SiLU(),
        )
        self.mlp_seq = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            SiLU(),
        )

    def forward(self, seq):
        seq = seq + self._attention(seq)
        seq = seq + torch.cat([
            self.mlp_main(seq[:, :1]),
            self.mlp_seq(seq[:, 1:]),
        ], 1)
        return seq

    def _attention(self, seq):
        B, NU, H = seq.shape
        seq = torch.cat([
            self.main_qkv_linear(self.layer_norm_main(seq[:, :1])),
            self.seq_qkv_linear(self.layer_norm_seq(seq[:, 1:])),
        ], 1)
        q, k, v = seq.view(B, NU, 3, self.num_heads * self.emb_size).transpose(0, 1).unbind(-2)
        attn = self.attn(q, k, v)[0].transpose(0, 1)
        assert attn.shape == (B, NU, H)
        return silu(attn)