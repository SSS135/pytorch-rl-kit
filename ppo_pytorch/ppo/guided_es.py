from copy import deepcopy

import numpy as np
import torch
import torch.autograd

from .ppo import PPO, TrainingData


class GES(PPO):
    def __init__(self, *args,
                 grad_buffer_len=32,
                 steps_per_update=1,
                 es_lr=0.01,
                 es_std=0.01,
                 es_blend=0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        assert self.num_actors == 1
        self.grad_buffer_len = grad_buffer_len
        self.steps_per_update = steps_per_update
        self.es_lr = es_lr
        self.es_std = es_std
        self.es_blend = es_blend

        # (k, n) grad history matrix
        self._grad_buffer = None
        self._grad_buffer_index = 0
        # up to 2 * P weight sets
        self._weights_to_test = []
        # up to 2 * P rewards
        self._es_rewards = []
        # (P, n) noise matrix
        self._noise = None
        self._orig_model_weights = deepcopy(list(self.model.state_dict().values()))

        self.value_loss_scale = 0

    def _step(self, prev_states, rewards, dones, cur_states) -> np.ndarray:
        reward = rewards[0] if rewards is not None else 0
        done = dones[0] if dones is not None else False

        actions = super()._step(prev_states, rewards, dones, cur_states)

        if len(self._es_rewards) != 0:
            ent = self.model.pd.entropy(torch.tensor(self.sample.probs[-1])).item() if len(self.sample.probs) != 0 else 0
            self._es_rewards[-1] += reward * self.reward_scale + self.entropy_reward_scale * ent

        if self._grad_buffer is not None and done:
            if len(self._weights_to_test) == 0:
                if len(self._es_rewards) != 0:
                    self._es_update()
                    self._es_rewards.clear()
                self._generate_weights_to_test()
            self._es_rewards.append(0)
            self._set_next_model_weights()

        return actions

    def _es_update(self):
        fitness = torch.tensor(self._es_rewards).view(self.steps_per_update, 2)
        fitness = fitness[:, 0] - fitness[:, 1]
        # (n) vector
        grad = self.es_lr / (2 * self.es_std * self.steps_per_update) * (fitness @ self._noise)
        # print('es grad', grad.pow(2).mean().sqrt(), fitness)
        weights = list(self.model.state_dict().values())
        w_lens = [w.numel() for w in weights]
        for curw, g, origw in zip(weights, grad.split(w_lens, 0), self._orig_model_weights):
            origw += g.view_as(origw)
            curw.copy_(origw)

        if self._do_log:
            self.logger.add_scalar('es grad rms', grad.pow(2).mean().sqrt(), self.frame)
            self.logger.add_scalar('es buffer rms', self._grad_buffer.pow(2).mean().sqrt(), self.frame)
            self.logger.add_scalar('es fitness rms', fitness.pow(2).mean().sqrt(), self.frame)

    def _process_sample_tensors(
            self, rewards, values_old, *args, **kwargs):
        values_old.data.fill_(0)
        return super()._process_sample_tensors(rewards, values_old, *args, **kwargs)

    def _ppo_update(self, data: TrainingData):
        cur_weights = deepcopy(list(self.model.state_dict().values()))
        for src, dst in zip(self._orig_model_weights, self.model.state_dict().values()):
            dst.copy_(src)
        super()._ppo_update(data)
        grads = []
        for prev, new in zip(self._orig_model_weights, self.model.state_dict().values()):
            grads.append(new - prev)
            new.copy_(prev)
        for src, dst in zip(cur_weights, self.model.state_dict().values()):
            dst.copy_(src)
        self._update_grad_buffer(grads)

    def _generate_weights_to_test(self):
        assert len(self._weights_to_test) == 0

        weights = list(self._orig_model_weights)
        w_lens = [w.numel() for w in weights]

        self._noise = self._get_noise()

        for noises in zip(*self._noise.split(w_lens, dim=1)):
            self._weights_to_test.append([w + e.view_as(w) for (e, w) in zip(noises, weights)])
            self._weights_to_test.append([w - e.view_as(w) for (e, w) in zip(noises, weights)])

    def _get_noise(self):
        """Generates (P, n) noise matrix"""
        buf = self._grad_buffer
        k, n = buf.shape
        full_space_noise = buf.new_zeros(self.steps_per_update, n).normal_()
        subspace_noise = buf.new_zeros(self.steps_per_update, k).normal_() @ buf
        subspace_noise /= subspace_noise.pow(2).mean().sqrt()
        full_space_noise *= self.es_std * self.es_blend
        subspace_noise *= self.es_std * (1 - self.es_blend)
        # print('noises full', full_space_noise.pow(2).mean().sqrt(), 'ss', subspace_noise.pow(2).mean().sqrt())
        noise = full_space_noise + subspace_noise
        return noise

    def _set_next_model_weights(self):
        weights = self._weights_to_test.pop(0)
        for src, dst in zip(weights, self.model.state_dict().values()):
            dst.copy_(src)

    def _update_grad_buffer(self, new_grads):
        new_grads = torch.cat([g.view(-1) for g in new_grads], dim=0)
        # print('ppo grad', new_grads.pow(2).mean().sqrt())
        if self._grad_buffer is None:
            self._grad_buffer = new_grads.new_zeros((self.grad_buffer_len, new_grads.numel()))
        self._grad_buffer[self._grad_buffer_index] = new_grads
        self._grad_buffer_index += 1
        if self._grad_buffer_index >= self.grad_buffer_len:
            self._grad_buffer_index = 0