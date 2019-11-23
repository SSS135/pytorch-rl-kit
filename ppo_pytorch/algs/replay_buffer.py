import threading
from collections import defaultdict
from typing import Dict

import torch
import random


class ReplayBuffer:
    def __init__(self, capacity):
        self._sample_capacity = capacity
        self._data: Dict[torch.Tensor] = None
        self._index = 0
        self._full_loop = False
        self._num_actors = None
        self._horizon_capacity = None
        self._lock = threading.RLock()

    def push(self, **sample):
        with self._lock:
            if self._data is None:
                self._init_data(sample)
            self._add_sample(sample)

    def _init_data(self, sample):
        self._num_actors = next(iter(sample.values())).shape[0]
        assert self._sample_capacity % self._num_actors == 0
        self._horizon_capacity = self._sample_capacity // self._num_actors
        # (buf_size, num_actors, *)
        self._data = {k: v.cpu().new_zeros((self._horizon_capacity, *v.shape)) for k, v in sample.items()}

    def _add_sample(self, sample):
        for name, value in sample.items():
            self._data[name][self._index] = sample[name].cpu()

        self._index += 1
        if self._index >= self._horizon_capacity:
            self._index = 0
            self._full_loop = True

    def sample(self, rollouts, horizon, end_sampling_factor=1.0):
        index = self._index
        len_horizon = self._len_horizon
        nu = end_sampling_factor ** (1 / rollouts)
        samples = defaultdict(list)
        for r in range(rollouts):
            start = random.randrange(0, max(1, int((len_horizon - horizon) * nu ** r)))
            start = (index - horizon - start) % (len_horizon - horizon)
            actor = random.randrange(0, self._num_actors)
            with self._lock:
                for name, value in self._data.items():
                    samples[name].append(self._data[name][start:start + horizon, actor])
        return {k: torch.stack(v, 1) for k, v in samples.items()}

    def get_last_samples(self, horizon):
        with self._lock:
            index = self._index
            len_horizon = self._len_horizon
            assert len_horizon >= horizon

            def loop_slice(x):
                start = index - horizon
                end = index
                if start >= 0:
                    return x[start:end]
                else:
                    return torch.cat([x[start:], x[:end]], 0)

            return {k: loop_slice(v) for k, v in self._data.items()}

    @property
    def _len_horizon(self):
        return self._horizon_capacity if self._full_loop else self._index

    def __len__(self):
        with self._lock:
            return self._len_horizon * self._num_actors


def test_replay_buffer():
    num_actors = 8
    num_samples = 2048
    capacity = 512 * num_actors
    rollouts = (1, 16, 128, 1024)
    horizon = (1, 64, 512)
    iters = 100

    states = torch.randn((num_samples, num_actors, 4, 10, 12))
    actions = torch.randn((num_samples, num_actors, 6))
    rewards = torch.randn((num_samples, num_actors))

    buffer = ReplayBuffer(capacity)

    def check_sample(sample, r, h):
        assert sample['states'].shape == (h, r, 4, 10, 12)
        assert sample['actions'].shape == (h, r, 6)
        assert sample['rewards'].shape == (h, r)

    for st, ac, r in zip(states, actions, rewards):
        buffer.push(states=st, actions=ac, rewards=r)
        r, h = random.choice(rollouts), random.choice(horizon)
        if len(buffer) >= h * num_actors:
            check_sample(buffer.get_last_samples(h), num_actors, h)

    for _ in range(iters):
        r, h = random.choice(rollouts), random.choice(horizon)
        check_sample(buffer.sample(r, h), r, h)
