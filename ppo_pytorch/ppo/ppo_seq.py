import pprint
from collections import namedtuple
from functools import partial

import gym.spaces
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from ..common import DecayLR, ValueDecay
from ..common.gae import calc_advantages, calc_returns
from ..common.multi_dataset import MultiDataset
from ..common.probability_distributions import DiagGaussianPd
from ..common.rl_base import RLBase
from ..models import Sega_CNNSeqActor
from ..models.heads import HeadOutput
from .ppo import PPO, TrainingData
from collections import namedtuple
from .ppo_qrnn import RNNData, PPO_QRNN


class PPO_Seq(PPO_QRNN):
    def __init__(self, *args, eval_seq_len=7, model_factory=Sega_CNNSeqActor, **kwargs):
        super().__init__(*args, model_factory=model_factory, **kwargs)
        self.eval_seq_len = eval_seq_len

    def _take_step(self, states, dones):
        if len(self._rnn_data.memory) == 0:
            mem = None
        else:
            mem = torch.cat(self._rnn_data.memory[-self.eval_seq_len:], 0)
            mem = Variable(mem, volatile=True)
        dones = torch.zeros(self.num_actors) if dones is None else torch.from_numpy(np.asarray(dones, np.float32))
        dones = Variable(dones.unsqueeze(0))
        if self.cuda_eval:
            dones = dones.cuda()
        states = Variable(states.unsqueeze(0), volatile=True)
        ac_out, next_mem = self.model(states, mem, dones)
        if len(self._rnn_data.memory) == 0:
            self._rnn_data.memory.append(next_mem.data.clone().fill_(0))
        self._rnn_data.memory.append(next_mem.data)
        self._rnn_data.dones.append(dones.data[0])
        return HeadOutput(ac_out.probs.squeeze(0), ac_out.state_values.squeeze(0))

    def _ppo_update(self, data):
        last_mem = self._rnn_data.memory[-self.eval_seq_len:]
        last_dones = self._rnn_data.dones[-self.eval_seq_len:]
        self._rnn_data = RNNData(self._rnn_data.memory[-self.horizon - 2:], self._rnn_data.dones[-self.horizon - 1:])
        super()._ppo_update(data)
        self._rnn_data = RNNData(last_mem, last_dones)

