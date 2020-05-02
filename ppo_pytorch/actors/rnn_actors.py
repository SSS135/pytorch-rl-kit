from ppo_pytorch.common.attr_dict import AttrDict

from .actors import FeatureExtractorBase, create_ppo_actor
import torch
from torch import nn
from ..common.qrnn import DenseQRNN, QRNN
import sru


def create_ppo_rnn_actor(observation_space, action_space, hidden_size=128, num_layers=2,
                            split_policy_value_network=False):
    assert len(observation_space.shape) == 1

    def fx_factory(): return RNNFeatureExtractor(
        observation_space.shape[0], hidden_size, num_layers)
    return create_ppo_actor(action_space, fx_factory, split_policy_value_network, is_recurrent=True)


class RNNFeatureExtractor(FeatureExtractorBase):
    def __init__(self, input_size: int, hidden_size=128, num_layers=2, **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        assert self.norm_factory is None
        self.model = sru.SRU(input_size, hidden_size, num_layers, rescale=True, use_tanh=True)
        # self.model = QRNN(input_size, hidden_size, num_layers)

    @property
    def output_size(self):
        return self.hidden_size

    def reset_weights(self):
        super().reset_weights()
        if hasattr(self.model, 'reset_parameters'):
            self.model.reset_parameters()

    def forward(self, input: torch.Tensor, memory: torch.Tensor, dones: torch.Tensor, logger=None, cur_step=None, **kwargs):
        # memory: (B, L, *) -> (L, B, *)
        rnn_kwargs = dict(reset_flags=dones) if isinstance(self.model, QRNN) or isinstance(self.model, DenseQRNN) else dict()
        x, memory = self.model(input, memory.transpose(0, 1).contiguous() if memory is not None else None, **rnn_kwargs)
        if logger is not None:
            logger.add_histogram(f'layer_{self.num_layers - 1}_output', x, cur_step)
            logger.add_histogram(f'memory', memory, cur_step)
        # x: (H, B, *)
        # memory: (B, L, *)
        return x, memory.transpose(0, 1)