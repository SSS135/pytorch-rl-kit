import copy
import pprint
from collections import namedtuple, OrderedDict
from functools import partial

import gym.spaces
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from ..common import DecayLR, ValueDecay
from ..common.gae import calc_advantages, calc_returns
from ..common.multi_dataset import MultiDataset
from ..common.probability_distributions import DiagGaussianPd
from ..common.rl_base import RLBase
from ..models import QRNNActorCritic
from ..models.heads import HeadOutput
from .ppo import PPO, TrainingData
from collections import namedtuple


RNNData = namedtuple('RNNData', 'memory, dones')


class PPO_QRNN(PPO):
    def __init__(self, observation_space, action_space,
                 model_factory=QRNNActorCritic,
                 *args, **kwargs):
        super().__init__(observation_space, action_space, model_factory=model_factory, *args, **kwargs)
        self._rnn_data = RNNData([], [])
        assert (self.horizon * self.num_actors) % self.batch_size == 0

    def _reorder_data(self, data) -> TrainingData:
        def reorder(input):
            if input is None:
                return None
            # input: (seq * num_actors, ...)
            # (seq, num_actors, ...)
            x = input.reshape(-1, self.num_actors, *input.shape[1:])
            # (num_actors * seq, ...)
            return x.transpose(0, 1).reshape(input.shape)

        data = data._asdict()
        data = [reorder(v) for v in data.values()]
        return TrainingData._make(data)

    def _take_step(self, states, dones):
        self.model.eval()

        mem = self._rnn_data.memory[-1] if len(self._rnn_data.memory) != 0 else None
        dones = torch.zeros(self.num_actors) if dones is None else torch.from_numpy(np.asarray(dones, np.float32))
        dones = dones.unsqueeze(0)
        dones = dones.to(self.device_eval)
        states = states.unsqueeze(0)
        ac_out, next_mem = self.model(states, mem, dones)

        if len(self._rnn_data.memory) == 0:
            self._rnn_data.memory.append(next_mem.clone().fill_(0))
        self._rnn_data.memory.append(next_mem)
        self._rnn_data.dones.append(dones[0])

        return HeadOutput(ac_out.probs.squeeze(0), ac_out.state_values.squeeze(0))

    def _ppo_update(self, data):
        self.model.train()
        # move model to cuda or cpu
        self.model = self.model.to(self.device_train)

        data = self._reorder_data(data)

        memory = torch.stack(self._rnn_data.memory[:-2], 0)  # (steps, layers, actors, hidden_size)
        memory = memory.permute(2, 0, 1, 3)  # (actors, steps, layers, hidden_size)
        memory = memory.reshape(-1, *memory.shape[2:]) # (actors * steps, layers, hidden_size)

        dones = self._rnn_data.dones[:-1] # (steps, actors)
        dones = torch.stack(dones, 0).transpose(0, 1).reshape(-1) # (actors * steps)

        self._rnn_data = RNNData(self._rnn_data.memory[-1:], [])

        # actor_switch_flags = torch.zeros(self.horizon)
        # actor_switch_flags[-1] = 1
        # actor_switch_flags = actor_switch_flags.repeat(self.num_actors)

        # (actors * steps, ...)
        data = (data.states.pin_memory() if self.device_train.type == 'cuda' else data.states,
                data.probs_old, data.values_old, data.actions, data.advantages,
                data.returns, memory, dones)
        # (actors, steps, ...)
        num_actors = self.num_actors
        if self.horizon > self.batch_size:
            assert self.horizon % self.batch_size == 0
            num_actors *= self.horizon // self.batch_size
            batches = num_actors
        else:
            batches = max(1, num_actors * self.horizon // self.batch_size)

        data = [x.reshape(num_actors, -1, *x.shape[1:]) for x in data]

        prev_model_dict = copy.deepcopy(self.model.state_dict())

        for ppo_iter in range(self.ppo_iters):
            actor_index_chunks = torch.randperm(num_actors).to(self.device_train).chunk(batches)
            for loader_iter, ids in enumerate(actor_index_chunks):
                # prepare batch data
                # (actors * steps, ...)
                st, po, vo, ac, adv, ret, mem, done = [
                    Variable(x[ids].to(self.device_train).reshape(-1, *x.shape[2:]))
                    for x in data]
                # (steps, actors, ...)
                st, mem, done = [x.reshape(ids.shape[0], -1, *x.shape[1:]).transpose(0, 1) for x in (st, mem, done)]
                # (layers, actors, hidden_size)
                mem = mem[0].transpose(0, 1)
                # (steps, actors)
                done = done.reshape(done.shape[:2])

                if ppo_iter == self.ppo_iters - 1 and loader_iter == 0:
                    self.model.set_log(self.logger, self._do_log, self.step)

                with torch.enable_grad():
                    actor_out, _ = self.model(st, mem, done)
                    # (actors * steps, probs)
                    probs = actor_out.probs.transpose(0, 1).reshape(-1, actor_out.probs.shape[2])
                    # (actors * steps)
                    state_values = actor_out.state_values.transpose(0, 1).reshape(-1)
                    # get loss
                    loss, kl = self._get_ppo_loss(probs, po, state_values, vo, ac, adv, ret)
                    loss = loss.mean()

                    # optimize
                    loss.backward()
                    clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                self.model.set_log(self.logger, False, self.step)

            if self._do_log and ppo_iter == self.ppo_iters - 1:
                self.logger.add_scalar('learning rate', self.learning_rate, self.frame)
                self.logger.add_scalar('clip mult', self.clip_mult, self.frame)
                self.logger.add_scalar('total loss', loss, self.frame)
                self.logger.add_scalar('kl', kl, self.frame)

    def drop_collected_steps(self):
        super().drop_collected_steps()
        self._rnn_data = RNNData([], [])
