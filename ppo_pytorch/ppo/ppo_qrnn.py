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
from ..models import QRNNActorCritic
from ..models.actors import ActorOutput
from .ppo import PPO, TrainingData
from collections import namedtuple


RNNData = namedtuple('RNNData', 'memory, dones')


class PPO_QRNN(PPO):
    def __init__(self, observation_space, action_space,
                 model_factory=QRNNActorCritic,
                 *args, **kwargs):
        super().__init__(observation_space, action_space, model_factory=model_factory, *args, **kwargs)
        self._rnn_data = RNNData([], [])

    def _reorder_data(self, data) -> TrainingData:
        def reorder(input):
            # input: (seq * num_actors, ...)
            # (seq, num_actors, ...)
            x = input.view(-1, self.num_actors, *input.shape[1:])
            # (num_actors * seq, ...)
            return x.transpose(0, 1).contiguous().view_as(input)

        data = data._asdict()
        data = [reorder(v) for v in data.values()]
        return TrainingData._make(data)

    def _take_step(self, states, dones):
        mem = Variable(self._rnn_data.memory[-1], volatile=True) if len(self._rnn_data.memory) != 0 else None
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
        return ActorOutput(ac_out.probs.squeeze(0), ac_out.state_values.squeeze(0))

    def _ppo_update(self, data):
        self.model.train()
        # move model to cuda or cpu
        if next(self.model.parameters()).is_cuda != self.cuda_train:
            self.model = self.model.cuda() if self.cuda_train else self.model.cpu()

        data = self._reorder_data(data)

        memory = torch.stack(self._rnn_data.memory[:-2], 0)  # (steps, layers, actors, hidden_size)
        memory = memory.permute(2, 0, 1, 3).contiguous()  # (actors, steps, layers, hidden_size)
        memory = memory.view(-1, *memory.shape[2:]) # (actors * steps, layers, hidden_size)

        dones = self._rnn_data.dones[:-1] # (steps, actors)
        dones = torch.stack(dones, 0).transpose(0, 1).contiguous().view(-1) # (actors * steps)

        self._rnn_data = RNNData(self._rnn_data.memory[-1:], [])

        # actor_switch_flags = torch.zeros(self.horizon)
        # actor_switch_flags[-1] = 1
        # actor_switch_flags = actor_switch_flags.repeat(self.num_actors)

        # (actors * steps, ...)
        data = (data.states, data.probs_old, data.values_old, data.actions, data.advantages,
                data.returns, memory, dones)
        # (actors, steps, ...)
        data = [x.view(self.num_actors, -1, *x.shape[1:]) for x in data]

        batches = max(1, self.num_actors * self.horizon // self.batch_size)

        for ppo_iter in range(self.ppo_iters):
            actor_index_chunks = torch.randperm(self.num_actors).chunk(batches)
            for loader_iter, ids in enumerate(actor_index_chunks):
                # prepare batch data
                # (actors * steps, ...)
                st, po, vo, ac, adv, ret, mem, done = [
                    Variable(x[ids.cuda() if x.is_cuda else ids].contiguous().view(-1, *x.shape[2:]))
                    for x in data]
                # (steps, actors, ...)
                st, mem, done = [x.data.view(ids.shape[0], -1, *x.shape[1:]).transpose(0, 1) for x in (st, mem, done)]
                # (layers, actors, hidden_size)
                mem = mem[0].transpose(0, 1)
                if self.cuda_train:
                    st, mem, done = [x.cuda() for x in (st, mem, done)]
                # (steps, actors)
                done = done.contiguous().view(done.shape[:2])
                st, mem, done = [Variable(x) for x in (st, mem, done)]

                if ppo_iter == self.ppo_iters - 1 and loader_iter == 0:
                    self.model.set_log(self.logger, self._do_log, self.step)
                actor_out, _ = self.model(st, mem, done)
                # (actors * steps, probs)
                probs = actor_out.probs.cpu().transpose(0, 1).contiguous().view(-1, actor_out.probs.shape[2])
                # (actors * steps)
                state_values = actor_out.state_values.cpu().transpose(0, 1).contiguous().view(-1)
                # get loss
                loss, kl = self._get_ppo_loss(probs, po, state_values, vo, ac, adv, ret)

                # optimize
                loss.backward()
                clip_grad_norm(self.model.parameters(), self.grad_clip_norm)
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

