from collections import namedtuple

import torch
from torch.nn.utils import clip_grad_norm_

from .ppo import PPO
from ..models import RNNActor
from ..models.heads import HeadOutput
from ..models.utils import image_to_float
from ..common.attr_dict import AttrDict


class PPO_RNN(PPO):
    def __init__(self, observation_space, action_space,
                 model_factory=RNNActor,
                 *args, **kwargs):
        super().__init__(observation_space, action_space, model_factory=model_factory, *args, **kwargs)
        assert (self.horizon * self.num_actors) % self.batch_size == 0
        self.layers = None

    def _reorder_data(self, data: AttrDict) -> AttrDict:
        def reorder(input):
            # input: (seq * num_actors, ...)
            # (seq, num_actors, ...)
            x = input.reshape(-1, self.num_actors, *input.shape[1:])
            # (num_actors * seq, ...)
            return x.transpose(0, 1).reshape(input.shape)

        return AttrDict({k: reorder(v) for k, v in data.items()})

    def _take_step(self, states, dones):
        self.model.eval()

        data = self._steps_processor.data
        # (layers, num_actors, hidden_size)
        mem = data.memory[-1] if 'memory' in data else None
        dones = torch.zeros(self.num_actors) if dones is None else dones
        dones = dones.unsqueeze(0).to(self.device_eval)
        states = states.unsqueeze(0)
        ac_out, next_mem = self.model(states, mem, dones)
        self.layers = next_mem.shape[0]

        if not self.disable_training:
            if 'memory' not in data:
                self._steps_processor.append_values(memory=next_mem.clone().fill_(0))
            self._steps_processor.append_values(memory=next_mem)

        return HeadOutput(probs=ac_out.probs.squeeze(0), state_values=ac_out.state_values.squeeze(0))

    def _ppo_update(self, data: AttrDict):
        self.model.train()
        # move model to cuda or cpu
        self.model = self.model.to(self.device_train)

        data = self._reorder_data(data)

        all_memory = data.memory.reshape(self.layers, -1, *data.memory.shape[1:])  # (layers, steps, actors, hidden_size)
        memory = all_memory[:, :-2].permute(2, 1, 0, 3)  # (actors, steps, layers, hidden_size)
        memory = memory.reshape(-1, *memory.shape[2:])  # (actors * steps, layers, hidden_size)

        dones = data.dones.reshape(self.num_actors, -1)[:, :-1].reshape(-1)  # (actors * steps)

        self._steps_processor.append_values(memory=all_memory[:, -1])

        print(data.states.shape, memory.shape, dones.shape)

        # actor_switch_flags = torch.zeros(self.horizon)
        # actor_switch_flags[-1] = 1
        # actor_switch_flags = actor_switch_flags.repeat(self.num_actors)

        # (actors * steps, ...)
        data = (data.states.pin_memory() if self.device_train.type == 'cuda' else data.states,
                data.probs, data.state_values, data.actions, data.advantages,
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

        for ppo_iter in range(self.ppo_iters):
            actor_index_chunks = torch.randperm(num_actors).to(self.device_train).chunk(batches)
            for loader_iter, ids in enumerate(actor_index_chunks):
                # prepare batch data
                # (actors * steps, ...)
                st, po, vo, ac, adv, ret, mem, done = [
                    x[ids].to(self.device_train).view(-1, *x.shape[2:])
                    for x in data]
                # (steps, actors, ...)
                st, mem, done = [x.reshape(ids.shape[0], -1, *x.shape[1:]).transpose(0, 1) for x in (st, mem, done)]
                # (layers, actors, hidden_size)
                mem = mem[0].transpose(0, 1)
                # (steps, actors)
                done = done.reshape(done.shape[:2])

                st, mem, done = (x.contiguous() for x in (st, mem, done))
                st = image_to_float(st)

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
                self._optimizer.step()
                self._optimizer.zero_grad()

                self.model.set_log(self.logger, False, self.step)

            if self._do_log and ppo_iter == self.ppo_iters - 1:
                self.logger.add_scalar('learning rate', self._learning_rate, self.frame)
                self.logger.add_scalar('clip mult', self._clip_mult, self.frame)
                self.logger.add_scalar('total loss', loss, self.frame)
                self.logger.add_scalar('kl', kl, self.frame)

