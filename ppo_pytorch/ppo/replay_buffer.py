from collections import defaultdict
from typing import Dict

import torch
import random


class ReplayBuffer:
    def __init__(self, capacity):
        self._capacity = capacity
        self._data: Dict[torch.Tensor] = None
        self._index = 0
        self._full_loop = False

    def push(self, **sample):
        if self._data is None:
            self._init_data(sample)
        self._add_sample(sample)

    def _init_data(self, sample):
        self._data = {k: v.cpu().new_zeros((self._capacity, *v.shape)) for k, v in sample.items()}

    def _add_sample(self, sample):
        for name, value in sample.items():
            self._data[name][self._index] = sample[name].cpu()

        self._index += 1
        if self._index >= self._capacity:
            self._index = 0
            self._full_loop = True

    def sample(self, rollouts, horizon):
        num_actors = next(iter(self._data.values())).shape[1]
        samples = defaultdict(list)
        for r in range(rollouts):
            start = random.randrange(0, max(1, len(self) - horizon))
            actor = random.randrange(0, num_actors)
            for name, value in self._data.items():
                samples[name].append(self._data[name][start:start + horizon, actor])
        return {k: torch.stack(v, 1) for k, v in samples.items()}

    def get_last_samples(self, horizon):
        return {k: v[self._index - horizon:self._index] for k, v in self._data.items()} # FIXME

    def __len__(self):
        return self._capacity if self._full_loop else self._index


def test_replay_buffer():
    num_actors = 8
    num_samples = 2048
    capacity = 512
    rollouts = (1, 16, 128, 1024)
    horizon = (1, 64, 512)
    iters = 100

    states = torch.randn((num_samples, num_actors, 4, 10, 12))
    actions = torch.randn((num_samples, num_actors, 6))
    rewards = torch.randn((num_samples, num_actors))

    buffer = ReplayBuffer(capacity)

    for st, ac, r in zip(states, actions, rewards):
        buffer.push(states=st, actions=ac, rewards=r)

    def check_sample(sample, r, h):
        assert sample['states'].shape == (h, r, 4, 10, 12)
        assert sample['actions'].shape == (h, r, 6)
        assert sample['rewards'].shape == (h, r)

    for _ in range(iters):
        r, h = random.choice(rollouts), random.choice(horizon)
        check_sample(buffer.sample(r, h), r, h)
        check_sample(buffer.get_last_samples(h), num_actors, h)