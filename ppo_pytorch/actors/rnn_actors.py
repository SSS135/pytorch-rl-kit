from ppo_pytorch.common.activation_norm import ActivationNorm
from ppo_pytorch.common.probability_distributions import make_pd
from ppo_pytorch.common.probability_distributions import ProbabilityDistribution
from ppo_pytorch.common.squash import squash
from torch.distributions import TanhTransform

from .actors import FeatureExtractorBase, create_ppo_actor, create_impala_actor
import torch
from torch import nn
from ..common.qrnn import DenseQRNN, QRNN
import sru
from torch import Tensor


def _create_rnn_actor(actor_fn, observation_space, action_space, hidden_size=128, num_layers=2,
                      split_policy_value_network=False, num_values=1, goal_size=0):
    assert len(observation_space.shape) == 1
    pd = make_pd(action_space)
    def fx_factory(): return RNNFeatureExtractor(
        pd, observation_space.shape[0], hidden_size, num_layers, goal_size=goal_size)
    return actor_fn(action_space, fx_factory, split_policy_value_network, num_values, is_recurrent=True)


def create_ppo_rnn_actor(*args, **kwargs):
    return _create_rnn_actor(create_ppo_actor, *args, **kwargs)


def create_impala_rnn_actor(*args, **kwargs):
    return _create_rnn_actor(create_impala_actor, *args, **kwargs)


class RNNFeatureExtractor(FeatureExtractorBase):
    def __init__(self, pd: ProbabilityDistribution, input_size: int,
                 hidden_size=128, num_layers=2, goal_size=0, env_input=True, **kwargs):
        super().__init__(**kwargs)
        self.pd = pd
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.goal_size = goal_size
        self.env_input = env_input
        assert self.norm_factory is None
        emb_size = hidden_size if env_input else input_size
        # self.model = sru.SRU(emb_size, hidden_size, num_layers, rescale=True)
        self.model = nn.LSTM(emb_size, hidden_size, num_layers)
        # self.model = nn.GRU(emb_size, hidden_size, num_layers)
        # self.model = QRNN(emb_size, hidden_size, num_layers)
        # self.model = DenseQRNN(emb_size, hidden_size, num_layers, output_gate=False)
        self.info_embedding = nn.Sequential(
            nn.Linear(goal_size + pd.input_vector_len + 1, 128),
            nn.Tanh(),
            nn.Linear(128, emb_size))
        self.input_embedding = nn.Sequential(
            nn.Linear(self.input_size, hidden_size),
            nn.Tanh())
        self.act_norm_x = ActivationNorm(1)
        self.act_norm_memory = ActivationNorm(1)

    @property
    def output_size(self):
        return self.hidden_size

    def reset_weights(self):
        super().reset_weights()
        if hasattr(self.model, 'reset_parameters'):
            self.model.reset_parameters()

    def forward(self, input: Tensor, memory: Tensor, dones: Tensor, prev_rewards: Tensor, prev_actions: Tensor,
                logger=None, cur_step=None, goal=None, **kwargs):
        if self.env_input:
            input = self.input_embedding(input)

        info_data = torch.cat([
            squash(prev_rewards.unsqueeze(-1)),
            self.pd.to_inputs(prev_actions),
            *((goal,) if self.goal_size != 0 else ())
        ], -1)
        info_mul = self.info_embedding(info_data)
        input = input * 2 * info_mul.sigmoid()

        rnn_kwargs = dict(reset_flags=dones) if isinstance(self.model, QRNN) or isinstance(self.model, DenseQRNN) else dict()

        # memory: (B, L, *) -> (L, B, *)
        if memory is not None:
            memory = memory.transpose(0, 1)
            if isinstance(self.model, nn.LSTM):
                memory = [m.contiguous() for m in memory.chunk(2, -1)]
            else:
                memory = memory.contiguous()

        x, memory = self.model(input, memory, **rnn_kwargs)

        # memory: (L, B, *) -> (B, L, *)
        if isinstance(self.model, nn.LSTM):
            memory = torch.cat(memory, -1)
        memory = memory.transpose(0, 1)

        self.act_norm_x(TanhTransform.atanh(x))
        # memory = self.act_norm_memory(memory)

        if logger is not None:
            logger.add_histogram(f'layer_{self.num_layers - 1}_output', x, cur_step)
            logger.add_histogram(f'memory', memory, cur_step)

        return x, memory