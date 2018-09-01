from collections import namedtuple

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from .ppo import PPO, TrainingData
from ..models import QRNNActor
from ..models.heads import HeadOutput
from ..models.utils import image_to_float

RNNData = namedtuple('RNNData', 'memory, dones')


class PPO_RNN(PPO):
    def __init__(self, observation_space, action_space,
                 model_factory=QRNNActor,
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
            x = input.contiguous().view(-1, self.num_actors, *input.shape[1:])
            # (num_actors * seq, ...)
            return x.transpose(0, 1).contiguous().view(input.shape)

        data = data._asdict()
        data = [reorder(v) for v in data.values()]
        return TrainingData._make(data)

    def _take_step(self, states, dones):
        self.model.eval()

        # (layers, batch_size, hidden_size)
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

        return HeadOutput(probs=ac_out.probs.squeeze(0), state_value=ac_out.state_value.squeeze(0))

    def _ppo_update(self, data):
        self.model.train()
        # move model to cuda or cpu
        self.model = self.model.to(self.device_train)

        data = self._reorder_data(data)

        memory = torch.stack(self._rnn_data.memory[:-2], 0)  # (steps, layers, actors, hidden_size)
        memory = memory.permute(2, 0, 1, 3)  # (actors, steps, layers, hidden_size)
        memory = memory.contiguous().view(-1, *memory.shape[2:]) # (actors * steps, layers, hidden_size)

        dones = self._rnn_data.dones[:-1] # (steps, actors)
        dones = torch.stack(dones, 0).transpose(0, 1).contiguous().view(-1) # (actors * steps)

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

        data = [x.contiguous().view(num_actors, -1, *x.shape[1:]) for x in data]

        for ppo_iter in range(self.ppo_iters):
            actor_index_chunks = torch.randperm(num_actors).to(self.device_train).chunk(batches)
            for loader_iter, ids in enumerate(actor_index_chunks):
                # prepare batch data
                # (actors * steps, ...)
                st, po, vo, ac, adv, ret, mem, done = [
                    x[ids].to(self.device_train).view(-1, *x.shape[2:])
                    for x in data]
                # (steps, actors, ...)
                st, mem, done = [x.contiguous().view(ids.shape[0], -1, *x.shape[1:]).transpose(0, 1) for x in (st, mem, done)]
                # (layers, actors, hidden_size)
                mem = mem[0].transpose(0, 1)
                # (steps, actors)
                done = done.contiguous().view(done.shape[:2])

                st, mem, done = (x.contiguous() for x in (st, mem, done))
                st = image_to_float(st)

                if ppo_iter == self.ppo_iters - 1 and loader_iter == 0:
                    self.model.set_log(self.logger, self._do_log, self.step)

                with torch.enable_grad():
                    actor_out, _ = self.model(st, mem, done)
                    # (actors * steps, probs)
                    probs = actor_out.probs.transpose(0, 1).contiguous().view(-1, actor_out.probs.shape[2])
                    # (actors * steps)
                    state_value = actor_out.state_value.transpose(0, 1).contiguous().view(-1)
                    # get loss
                    loss, kl = self._get_ppo_loss(probs, po, state_value, vo, ac, adv, ret)
                    # loss_vat = get_vat_loss(
                    #     lambda x: self.model(x.view_as(st), mem, done)[0].probs.view_as(probs),
                    #     st.view(-1, *st.shape[2:]),
                    #     actor_out.probs.view_as(probs),
                    #     custom_kl=lambda o, n: self.model.pd.kl(o, n).mean())
                    loss = loss.mean()

                # optimize
                loss.backward()
                if self.grad_clip_norm is not None:
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

