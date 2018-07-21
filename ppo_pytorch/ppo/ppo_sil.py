from collections import namedtuple

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from .ppo import PPO, TrainingData
from ..models import QRNNActor
from ..models.heads import HeadOutput
from ..models.utils import image_to_float
from .mppo import ReplayBuffer


class PPO_SIL(PPO):
    def __init__(self,
                 per_actor_replay_buffer_size=2 * 1024,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.per_actor_replay_buffer_size = per_actor_replay_buffer_size
        self.replay_buffer = ReplayBuffer(per_actor_replay_buffer_size)

    def _ppo_update(self, data):
        super()._ppo_update(data)

        # H x B x *
        states, actions, rewards, dones = [x.view(-1, self.num_actors, *x.shape[1:])
                                           for x in (data.states, data.actions, data.rewards, data.dones)]
        self.replay_buffer.push(states, actions, rewards, dones)

        # move model to cuda or cpu
        self.world_gen = self.world_gen.to(self.device_train).train()
        self.model = self.model.to(self.device_train).train()
        self.memory_init_model = self.memory_init_model.to(self.device_train).train()

        horizon = self.horizon # np.random.randint(2, 8)
        rollouts = self.ppo_iters

        # (H, B, ...)
        all_states, all_actions, all_rewards, all_dones = self.replay_buffer.sample(rollouts, horizon)
        all_dones = all_dones.astype(np.float32)

        data = [torch.from_numpy(x) for x in (all_states, all_actions, all_rewards, all_dones)]
        if self.device_train.type == 'cuda':
            data = [x.pin_memory() for x in data]

        for train_iter in range((rollouts * horizon) // self.batch_size):
            slc = (slice(None), slice(train_iter * rollouts, (train_iter + 1) * rollouts))
            # (H, B, ...)
            states, actions, rewards, dones = [x[slc] for x in data]
            states = states.to(self.device_train)
            actor_out = self.model(states)
