from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from .ppo_rnn import PPO_RNN
from ..common.probability_distributions import DiagGaussianPd
from ..actors import HRNNActor

RNNData = namedtuple('RNNData', 'memory, dones, action_l2, cur_l1, probs_l2, values_l2')


class PPO_HRNN(PPO_RNN):
    def __init__(self, observation_space, action_space, *args,
                 model_factory=HRNNActor,
                 reward_discount_l1=0.95,
                 advantage_discount_l1=0.95,
                 env_reward_scale_l1=1.0,
                 h_reward_scale_l1=0.1,
                 **kwargs):
        super().__init__(observation_space, action_space, model_factory=model_factory, *args, **kwargs)
        self._rnn_data = RNNData([], [], [], [], [], [])
        self.data_l2 = None
        self.reward_discount_l1 = reward_discount_l1
        self.advantage_discount_l1 = advantage_discount_l1
        self.env_reward_scale_l1 = env_reward_scale_l1
        self.h_reward_scale_l1 = h_reward_scale_l1
        assert (self.horizon * self.num_actors) % self.batch_size == 0

    def _take_step(self, states, dones):
        mem = self._rnn_data.memory[-1] if len(self._rnn_data.memory) != 0 else None
        dones = torch.zeros(self.num_actors) if dones is None else torch.from_numpy(np.asarray(dones, np.float32))
        dones = dones.unsqueeze(0).to(self.device_eval)
        states = states.unsqueeze(0)
        head_l1, head_l2, action_l2, cur_l1, next_mem = self.model(states, mem, dones)

        head_l1.probs = head_l1.probs.squeeze(0)
        head_l1.state_values = head_l1.state_values.squeeze(0)
        head_l2.probs = head_l2.probs.squeeze(0)
        head_l2.state_values = head_l2.state_values.squeeze(0)

        if len(self._rnn_data.memory) == 0:
            self._rnn_data.memory.append(next_mem.data.clone().fill_(0))
        self._rnn_data.memory.append(next_mem.data)
        self._rnn_data.dones.append(dones.data[0])
        self._rnn_data.action_l2.append(action_l2.data[0])
        self._rnn_data.cur_l1.append(cur_l1.data[0])
        self._rnn_data.probs_l2.append(head_l2.probs.data)
        self._rnn_data.values_l2.append(head_l2.state_values.data)

        return head_l1

    def get_l1_l2_samples(self):
        target = torch.stack(self._rnn_data.action_l2[:-1], 0)
        real = torch.stack(self._rnn_data.cur_l1[1:], 0) - torch.stack(self._rnn_data.cur_l1[:-1], 0)
        h_rewards_l1 = F.cosine_similarity(real[-target.shape[0]:], target, dim=-1)

        probs_l2 = torch.stack(self._rnn_data.probs_l2, 0).cpu().numpy()
        values_l2 = torch.stack(self._rnn_data.values_l2, 0).cpu().numpy()
        actions_l2 = torch.stack(self._rnn_data.action_l2, 0).cpu().numpy()

        sample_l1 = self.sample
        total_rewards_l2 = self.reward_scale * np.asarray(sample_l1.rewards)
        total_rewards_l1 = self.h_reward_scale_l1 * h_rewards_l1.cpu().numpy() + \
                           self.env_reward_scale_l1 * self.reward_scale * np.asarray(sample_l1.rewards)
        sample_l2 = Sample(None, total_rewards_l2, self.sample.dones, probs_l2, values_l2, actions_l2)
        sample_l1 = sample_l1._replace(rewards=total_rewards_l1)

        return sample_l1, sample_l2

    def _process_sample(self, *args, **kwargs):
        self.sample, sample_l2 = self.get_l1_l2_samples()
        self.data_l2 = super()._process_sample(
            sample_l2, self.model.h_pd, self.reward_discount, self.advantage_discount, 1)

        if self._do_log:
            self.logger.add_scalar('rewards l1', np.mean(self.sample.rewards), self.frame)
            self.logger.add_histogram('rewards l2', self.data_l2.rewards, self.frame)
            self.logger.add_histogram('returns l2', self.data_l2.returns, self.frame)
            self.logger.add_histogram('advantages l2', self.data_l2.advantages, self.frame)
            self.logger.add_histogram('values l2', self.data_l2.values_old, self.frame)
            if isinstance(self.model.h_pd, DiagGaussianPd):
                mean, std = self.data_l2.probs_old.chunk(2, dim=1)
                self.logger.add_histogram('probs mean l2', mean, self.frame)
                self.logger.add_histogram('probs std l2', std, self.frame)
            else:
                self.logger.add_histogram('probs l2', F.log_softmax(self.data_l2.probs_old, dim=-1), self.frame)

        return super()._process_sample(self.sample, self.model.pd, self.reward_discount_l1, self.advantage_discount_l1, 1)

    def _ppo_update(self, data_l1):
        self.model.train()
        # move model to cuda or cpu
        self.model = self.model.to(self.device_train)

        data_l1 = self._reorder_data(data_l1)
        data_l2 = self._reorder_data(self.data_l2)
        self.data_l2 = None

        memory = torch.stack(self._rnn_data.memory[:-2], 0)  # (steps, layers, actors, hidden_size)
        memory = memory.permute(2, 0, 1, 3)  # (actors, steps, layers, hidden_size)
        memory = memory.reshape(-1, *memory.shape[2:]) # (actors * steps, layers, hidden_size)

        dones = self._rnn_data.dones[:-1] # (steps, actors)
        dones = torch.stack(dones, 0).transpose(0, 1).reshape(-1) # (actors * steps)

        # (steps, actors, state_size)
        state_diff = torch.stack(self._rnn_data.cur_l1[1:], 0) - torch.stack(self._rnn_data.cur_l1[:-1], 0)
        state_diff = state_diff[-self.horizon:]
        state_diff = state_diff.transpose(0, 1).reshape(-1, state_diff.shape[-1])
        state_diff /= state_diff.abs().max(-1, keepdim=True)[0].clamp(1e-5, 1e8)
        assert ((state_diff < -1) | (state_diff > 1)).sum() == 0

        self._rnn_data = RNNData(self._rnn_data.memory[-1:], [], [], self._rnn_data.cur_l1[-1:], [], [])

        # rewards, returns, advantages = self._process_rewards(
        #     data_l2.rewards, data_l1values_old, dones, reward_discount, advantage_discount, reward_scale)
        #
        # advantages_l1 /= (1 + self.real_reward_l1_frac)
        # (actors * steps, ...)
        data = (data_l1.states.pin_memory() if self.device_train.type == 'cuda' else data_l1.states,
                data_l1.probs_old, data_l2.probs_old,
                data_l1.values_old, data_l2.values_old,
                data_l1.actions, data_l2.actions,
                data_l1.advantages, data_l2.advantages,
                data_l1.returns, data_l2.returns,
                state_diff, memory, dones)
        # (actors, steps, ...)
        num_actors = self.num_actors
        if self.horizon > self.batch_size:
            assert self.horizon % self.batch_size == 0
            num_actors *= self.horizon // self.batch_size
            batches = num_actors
        else:
            batches = max(1, num_actors * self.horizon // self.batch_size)

        data = [x.reshape(num_actors, -1, *x.shape[1:]).to(self.device_train) for x in data]

        initial_lr = [g['lr'] for g in self._optimizer.param_groups]

        for ppo_iter in range(self.ppo_iters):
            actor_index_chunks = torch.randperm(num_actors).to(self.device_train).chunk(batches)
            for loader_iter, ids in enumerate(actor_index_chunks):
                # prepare batch data
                # (actors * steps, ...)
                st, po_l1, po_l2, vo_l1, vo_l2, ac_l1, ac_l2, adv_l1, adv_l2, ret_l1, ret_l2, st_d, mem, done = [
                    x[ids].reshape(-1, *x.shape[2:])
                    for x in data]
                # (steps, actors, ...)
                st, mem, done, ac_l2_inp = [x.data.view(ids.shape[0], -1, *x.shape[1:]).transpose(0, 1)
                                           for x in (st, mem, done, ac_l2)]
                # (layers, actors, hidden_size)
                mem = mem[0].transpose(0, 1)
                # (steps, actors)
                done = done.reshape(done.shape[:2])

                if ppo_iter == self.ppo_iters - 1 and loader_iter == 0:
                    self.model.set_log(self.logger, self._do_log, self.step)

                loss, kl_l1, kl_l2 = self._hrnn_step(st, po_l1, po_l2, vo_l1, vo_l2, ac_l1, ac_l2_inp,
                                                     adv_l1, adv_l2, ret_l1, ret_l2, st_d, mem, done)

                # optimize
                loss.backward()
                if self.grad_clip_norm is not None:
                    clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self._optimizer.step()
                self._optimizer.zero_grad()

                self.model.set_log(self.logger, False, self.step)

            if self._do_log and ppo_iter == self.ppo_iters - 1:
                self.logger.add_scalar('learning rate', self._learning_rate, self.frame)
                self.logger.add_scalar('clip mult', self.policy_clip, self.frame)
                self.logger.add_scalar('total loss', loss, self.frame)
                self.logger.add_scalar('kl', kl_l1, self.frame)
                self.logger.add_scalar('kl_l2', kl_l2, self.frame)

            for g in self._optimizer.param_groups:
                g['lr'] *= self.lr_iter_mult

        for g, lr in zip(self._optimizer.param_groups, initial_lr):
            g['lr'] = lr

    def _hrnn_step(self, st, po_l1, po_l2, vo_l1, vo_l2, ac_l1, ac_l2_inp, adv_l1, adv_l2, ret_l1, ret_l2, st_d, mem, done):
        with torch.enable_grad():
            actor_out_l1, actor_out_l2, _, _, _ = self.model(st, mem, done, ac_l2_inp)
            # (actors * steps, probs)
            probs_l1, probs_l2 = [h.probs.transpose(0, 1).reshape(-1, h.probs.shape[2])
                                  for h in (actor_out_l1, actor_out_l2)]
            # (actors * steps)
            state_value_l1, state_value_l2 = [h.state_values.transpose(0, 1).reshape(-1)
                                              for h in (actor_out_l1, actor_out_l2)]

            # get loss
            loss_l1, kl_l1 = self._get_ppo_loss(probs_l1, po_l1, state_value_l1, vo_l1, ac_l1, adv_l1, ret_l1,
                                                self.model.pd, tag='')
            loss_l2, kl_l2 = self._get_ppo_loss(probs_l2, po_l2, state_value_l2, vo_l2, st_d, adv_l2, ret_l2,
                                                self.model.h_pd, tag=' l2')
            loss = loss_l1.mean() + loss_l2.view(self.num_actors, -1)[:, :-1].mean()

            return loss, kl_l1, kl_l2


    def drop_collected_steps(self):
        super().drop_collected_steps()
        self._rnn_data = RNNData([], [], [], [], [], [])

