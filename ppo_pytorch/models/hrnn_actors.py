import gym
import gym.spaces
import numpy as np
import torch
import torch.nn as nn
from ..common.qrnn import DenseQRNN
from optfn.temporal_group_norm import TemporalGroupNorm
from torch.autograd import Variable

from .actors import Actor
from .cnn_actors import CNNActor
from .utils import image_to_float
from ..common.probability_distributions import make_pd, BernoulliPd, DiagGaussianPd, FixedStdGaussianPd, BetaPd
from .rnn_actors import RNNActor


# class Sega_CNN_HQRNNActor(CNN_QRNNActor):
#     def __init__(self, *args, h_action_size=128, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.h_action_size = h_action_size
#
#         self.h_action_space = gym.spaces.Box(-1, 1, h_action_size)
#         self.h_observation_space = gym.spaces.Box(-1, 1, self.hidden_code_size)
#         self.h_pd = make_pd(self.h_action_space)
#         # self.gate_action_space = gym.spaces.Discrete(2)
#         # self.gate_pd = make_pd(self.gate_action_space)
#
#         layer_norm = self.norm is not None and 'layer' in self.norm
#
#         # self.qrnn_l1 = self.qrnn
#         del self.qrnn
#         # self.qrnn_l2 = DenseQRNN(self.hidden_code_size + h_action_size, self.hidden_code_size, self.num_layers, layer_norm=layer_norm)
#         self.qrnn_l1 = nn.Sequential(
#             nn.Linear(1920, self.hidden_code_size),
#             nn.ReLU(),
#         )
#         self.qrnn_l2 = nn.Sequential(
#             nn.Linear(self.hidden_code_size + h_action_size, self.hidden_code_size),
#             nn.ReLU(),
#         )
#         # self.action_upsample_l2 = nn.Sequential(
#         #     nn.Linear(h_action_size, self.hidden_code_size, bias=not layer_norm),
#         #     *([TemporalLayerNorm1(self.hidden_code_size)] if layer_norm else []),
#         #     # nn.ReLU(),
#         # )
#         self.action_merge_l1 = nn.Sequential(
#             nn.Linear(h_action_size * 2, self.hidden_code_size, bias=not layer_norm),
#             *([TemporalGroupNorm(1, self.hidden_code_size)] if layer_norm else []),
#             nn.ReLU(),
#         )
#         self.state_vec_extractor_l1 = nn.Sequential(
#             nn.Linear(self.hidden_code_size, h_action_size),
#             TemporalGroupNorm(1, h_action_size, affine=False),
#         )
#         self.action_l2_norm = TemporalGroupNorm(1, h_action_size, affine=False)
#         # self.norm_action_l2 = LayerNorm1d(self.hidden_code_size, affine=False)
#         # self.norm_hidden_l1 = LayerNorm1d(self.hidden_code_size, affine=False)
#         # self.head_gate_l2 = ActorCriticHead(self.hidden_code_size, self.gate_pd)
#         self.head_l2 = ActorCriticHead(self.hidden_code_size, self.h_pd)
#         self.reset_weights()
#
#     def extract_l1_features(self, input):
#         seq_len, batch_len = input.shape[:2]
#         input = input.contiguous().view(seq_len * batch_len, *input.shape[2:])
#
#         input = image_to_float(input)
#         x = self._extract_features(input)
#         x = x.view(seq_len, batch_len, -1)
#         hidden_l1 = self.qrnn_l1(x)
#         return hidden_l1
#
#     def act_l1(self, hidden_l1, target_l1):
#         x = torch.cat([hidden_l1, target_l1], -1)
#         x = self.action_merge_l1(x)
#         return x
#
#     def forward(self, input, memory, done_flags, action_l2=None):
#         # memory_l1, memory_l2 = memory.chunk(2, 0) if memory is not None else (None, None)
#
#         hidden_l1 = self.extract_l1_features(input)
#         state_vec_l1 = self.state_vec_extractor_l1(hidden_l1)
#         input_l2 = torch.cat([hidden_l1, state_vec_l1], -1)
#         # gate_l2 = self.head_gate_l2(hidden_l1)
#         hidden_l2 = self.qrnn_l2(input_l2)
#
#         head_l2 = self.head_l2(hidden_l2)
#         if action_l2 is None:
#             action_l2 = self.h_pd.sample(head_l2.probs)
#         target_l2 = self.action_l2_norm(action_l2 + state_vec_l1).detach()
#
#         preact_l1 = self.act_l1(state_vec_l1, target_l2)
#
#         head_l1 = self.head(preact_l1)
#         # head_l1.state_value = 0 * head_l1.state_value
#
#         next_memory = Variable(input.new(2, input.shape[1], 2))
#
#         return head_l1, head_l2, action_l2, state_vec_l1, target_l2, next_memory


class HRNNActor(RNNActor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.h_action_size = 16

        obs_len = int(np.product(self.observation_space.shape))

        self.h_pd = BetaPd(self.h_action_size, 1.01)
        # self.gate_pd = BernoulliPd(1)

        self.rnn_l1 = DenseQRNN(self.h_action_size * 2, self.hidden_code_size, self.num_layers, norm=self.norm)
        del self.rnn
        self.rnn_l2 = DenseQRNN(self.h_action_size, self.hidden_code_size, self.num_layers, norm=self.norm)

        # self.action_upsample_l2 = nn.Sequential(
        #     nn.Linear(h_action_size, self.hidden_code_size, bias=not layer_norm),
        #     *([TemporalLayerNorm1(self.hidden_code_size)] if layer_norm else []),
        #     # nn.ReLU(),
        # )
        self.input_emb_l2 = nn.Sequential(
            nn.Linear(obs_len, self.h_action_size),
            nn.Tanh(),
            # TemporalLayerNorm(self.h_action_size),
            # nn.ReLU(),
        )
        self.input_emb_l1 = nn.Sequential(
            nn.Linear(obs_len, self.h_action_size),
            nn.Tanh(),
            # TemporalLayerNorm(self.h_action_size),
            # nn.ReLU(),
        )
        # self.action_merge_l1 = nn.Sequential(
        #     nn.Linear(self.hidden_code_size * 2, self.hidden_code_size),
        #     nn.Tanh(),
        #     nn.Linear(self.hidden_code_size, self.hidden_code_size),
        #     nn.Tanh(),
        # )
        # self.target_emb_l2 = nn.Sequential(
        #     nn.Linear(self.h_action_size, self.hidden_code_size),
        #     # TemporalGroupNorm(8, self.hidden_code_size),
        #     # nn.ReLU(),
        # )
        # self.state_vec_extractor_l1 = nn.Sequential(
        #     nn.Linear(self.hidden_code_size, h_action_size),
        #     # TemporalGroupNorm(1, h_action_size, affine=False),
        # )
        # self.action_l2_norm = TemporalGroupNorm(1, h_action_size, affine=False)
        # self.norm_cur_l1 = TemporalLayerNorm(h_action_size, elementwise_affine=False)
        # self.norm_target_l1 = TemporalLayerNorm(h_action_size, elementwise_affine=False)
        # self.norm_hidden_l1 = LayerNorm1d(self.hidden_code_size, affine=False)
        self.heads_l2 = self._create_heads('heads_l2', self.hidden_code_size, self.h_pd, self.head_factory)
        # for name, head in self.heads_l2.items():
        #     # for p in head.parameters():
        #     #     p.requires_grad = False
        #     self.add_module(f'heads_l2_{name}', head)
        # self.head_gate_l2 = ActorCriticHead(self.hidden_code_size, self.gate_pd, math.log(0.2))
        self.reset_weights()

    def forward(self, input, memory, done_flags, action_l2=None):
        memory_l1, memory_l2 = memory.chunk(2, 0) if memory is not None else (None, None)

        state_l2 = self.input_emb_l2(input)
        # input_emb_l2 = input_emb_l2 / input_emb_l2.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
        # cur_l1 = hidden_l1 = input #= F.layer_norm(hidden_l1, hidden_l1.shape[-1:])
        # head_gate_l2 = self.head_gate_l2(hidden_l1)
        # if action_gate_l2 is None:
        #     # (actors, batch, 1)
        #     action_gate_l2 = self.gate_pd.sample(head_gate_l2.probs)
        hidden_l2, next_memory_l2 = self.rnn_l2(state_l2, memory_l2, done_flags)
        # hidden_l2 = F.layer_norm(hidden_l2, hidden_l2.shape[-1:])
        # cur_l1 = self.state_vec_extractor_l1(hidden_l1)
        # cur_l1 = hidden_l1

        head_l2 = self._run_heads(hidden_l2, self.heads_l2)
        if action_l2 is None:
            action_l2 = self.h_pd.sample(head_l2.probs)
            # action_l2 = action_l2 / action_l2.pow(2).mean(-1, keepdim=True).sqrt()
            action_l2 = action_l2 / action_l2.abs().max(-1, keepdim=True)[0]
            action_l2 = action_l2.detach()
        # target_l1 = (state_l2 + action_l2).detach()
        # cur_l1_norm = F.layer_norm(cur_l1, cur_l1.shape[-1:])
        # target_l1_norm = F.layer_norm(target_l1, target_l1.shape[-1:])

        input_emb_l1 = self.input_emb_l1(input)
        # input_emb_l1 = input_emb_l1 / input_emb_l1.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
        input_l1 = torch.cat([input_emb_l1, action_l2], -1)
        hidden_l1, next_memory_l1 = self.rnn_l1(input_l1, memory_l1, done_flags)
        # preact_l1 = torch.cat([hidden_l1, self.target_emb(target_l1)], -1)
        # preact_l1 = self.action_merge_l1(preact_l1)
        head_l1 = self._run_heads(hidden_l1, self.heads)

        next_memory = torch.cat([next_memory_l1, next_memory_l2], 0)
        # head_l1.state_value = head_l1.state_value * 0

        return head_l1, head_l2, action_l2, state_l2, next_memory