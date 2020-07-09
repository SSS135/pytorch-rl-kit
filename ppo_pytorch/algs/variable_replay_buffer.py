import threading
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Collection

import torch
import random
from torch import Tensor


DONE = 'dones'


class BufferThread:
    def __init__(self, capacity, horizon, num_burn_in_samples, end_sampling_factor):
        self.capacity = capacity
        self.horizon = horizon
        self.num_burn_in_samples = num_burn_in_samples
        self.end_sampling_factor = end_sampling_factor
        self._num_new_samples = 0
        self._data: Dict[str, Tensor] = None
        self._index = 0
        self._full_loop = False

    @property
    def avail_new_samples(self):
        samples = self._num_new_samples + self.num_burn_in_samples
        if len(self) < self.horizon or samples < self.horizon:
            return 0
        return samples // self.horizon * self.horizon

    def reduce_capacity(self, new_capacity):
        assert new_capacity <= self.capacity
        if self.capacity == new_capacity:
            return
        cut_end = max(self._index, new_capacity)
        cut_start = cut_end - new_capacity
        slc = slice(cut_start, cut_end)
        self._data = {k: v[slc].contiguous() for k, v in self._data.items()}
        self.capacity = new_capacity
        self._check_full_loop()

    def push(self, sample: Collection[Tuple[str, Tensor]]):
        if self._data is None:
            self._init_data(sample)
        self._add_sample(sample)

    def sample(self, cur_rollout, total_rollouts) -> Dict[str, Tensor]:
        assert 0 <= cur_rollout < total_rollouts and total_rollouts > 0
        index = self._index
        buf_len = len(self)
        end_sampling_factor = min(1.0, self.end_sampling_factor * self.capacity / len(self))
        nu = end_sampling_factor ** (1 / total_rollouts)
        end_inv = random.randrange(0, max(1, int((buf_len - self.horizon) * nu ** cur_rollout)))
        start = min((index - self.horizon - end_inv) % buf_len, buf_len - self.horizon)
        end = start + self.horizon
        assert start >= 0 and end >= 0, (start, end)
        assert start + self.horizon <= buf_len and end <= buf_len, (start, end, self.horizon, buf_len)
        assert self._full_loop or end <= index
        # [H, *]
        return {k: v[start:end] for k, v in self._data.items()}

    def get_new_samples(self) -> Dict[str, Tensor]:
        index = self._index - self.num_burn_in_samples
        self._num_new_samples += self.num_burn_in_samples
        num_samples = self._num_new_samples // self.horizon * self.horizon
        assert 0 < num_samples < self.capacity
        self._num_new_samples = max(0, self._num_new_samples - num_samples)

        start = index - num_samples - self._num_new_samples
        end = start + num_samples
        if start < -num_samples:
            start += self.capacity
            end += self.capacity
        if end > self.capacity:
            start -= self.capacity
            end -= self.capacity

        def loop_slice(x):
            if start >= 0:
                x = x[start:end]
            else:
                x = torch.cat([x[start:], x[:end]], 0)
            # [H, B, *]
            return x.view(num_samples // self.horizon, self.horizon, *x.shape[1:]).transpose(0, 1)

        return {k: loop_slice(v) for k, v in self._data.items()}

    def __len__(self):
        return self.capacity if self._full_loop else self._index

    def _init_data(self, sample: Collection[Tuple[str, Tensor]]):
        # (buf_size, *)
        self._data = {k: v.new_zeros((self.capacity, *v.shape)) for k, v in sample}

    def _add_sample(self, sample: Collection[Tuple[str, Tensor]]):
        for name, value in sample:
            self._data[name][self._index] = value

        self._num_new_samples += 1
        self._index += 1
        self._check_full_loop()

    def _check_full_loop(self):
        if self._index >= self.capacity:
            self._index = 0
            self._full_loop = True


class VariableReplayBuffer:
    def __init__(self, capacity, horizon, num_burn_in_samples, end_sampling_factor=1.0):
        self._total_capacity = capacity
        self.horizon = horizon
        self.num_burn_in_samples = num_burn_in_samples
        self.end_sampling_factor = end_sampling_factor
        self._buffers: List[BufferThread] = []
        self._actor_id_to_buf_index: Dict[int, int] = {}
        self._lock = threading.Lock()

    @property
    def avail_new_samples(self):
        with self._lock:
            return sum(buf.avail_new_samples for buf in self._buffers)

    def push(self, actor_ids: Tensor, **sample: Tensor):
        with self._lock:
            assert actor_ids.shape == sample[DONE].shape and actor_ids.dtype == torch.long, actor_ids

            sample_aids = actor_ids.tolist()
            assert len(sample_aids) == len(set(sample_aids)), sample_aids

            # resize buffers
            buf_aid_set = set(self._actor_id_to_buf_index.keys())
            sample_aid_set = set(sample_aids)
            desired_buf_count = len(buf_aid_set | sample_aid_set)
            if desired_buf_count > len(self._buffers):
                self._increase_num_buffers(desired_buf_count)

            # fill new actor ids to buffer
            new_aids = sample_aid_set - buf_aid_set
            free_buffer_pointers = list(set(range(len(self._buffers))) - set(self._actor_id_to_buf_index.values()))
            free_buffer_pointers.sort(key=lambda idx: -len(self._buffers[idx]))
            for new_aid in new_aids:
                pointer = free_buffer_pointers.pop()
                assert new_aid not in self._actor_id_to_buf_index and pointer not in self._actor_id_to_buf_index.values()
                self._actor_id_to_buf_index[new_aid] = pointer

            # add samples
            sample_items = list(sample.items())
            for i, aid in enumerate(sample_aids):
                buffer = self._buffers[self._actor_id_to_buf_index[aid]]
                buffer.push([(k, v[i]) for k, v in sample_items])

            # remove done actors
            dones = sample[DONE].tolist()
            for ac_sample_pos, (sample_aid, ac_done) in enumerate(zip(sample_aids, dones)):
                if ac_done and sample_aid in self._actor_id_to_buf_index:
                    del self._actor_id_to_buf_index[sample_aid]

    def sample(self, num_rollouts) -> Dict[str, Tensor]:
        with self._lock:
            # calc buffer index for each rollout
            buffer_chances = torch.tensor([len(buf) for buf in self._buffers], dtype=torch.float)
            buffer_chances[buffer_chances < self.horizon] = 0
            rollout_buffer_indices = torch.multinomial(buffer_chances, num_rollouts, replacement=True).tolist()

            rollouts = []
            for rollout_index, buf_index in enumerate(rollout_buffer_indices):
                buffer = self._buffers[buf_index]
                rollouts.append(buffer.sample(rollout_index, num_rollouts))
            # [H, B, *]
            return {k: torch.stack([r[k] for r in rollouts], 1) for k in rollouts[0].keys()}

    def get_new_samples(self):
        with self._lock:
            # [H, ~B, *]
            rollouts = [buf.get_new_samples() for buf in self._buffers if buf.avail_new_samples > 0]
            # [H, B, *]
            return {k: torch.cat([r[k] for r in rollouts], 1) for k in rollouts[0].keys()}

    def __len__(self):
        with self._lock:
            return sum(map(len, self._buffers))

    def _increase_num_buffers(self, new_num_buf):
        assert new_num_buf >= len(self._buffers)
        new_cap = self._total_capacity // new_num_buf
        for buf in self._buffers:
            buf.reduce_capacity(new_cap)
        while len(self._buffers) < new_num_buf:
            self._buffers.append(BufferThread(new_cap, self.horizon, self.num_burn_in_samples, self.end_sampling_factor))


def test_variable_replay_buffer_normal():
    torch.manual_seed(123)
    random.seed(123)

    num_actors = 8
    capacity = 512 * num_actors
    horizon = 64
    num_gs = 10
    iter_len = 512
    actor_ids = torch.arange(num_actors)

    buffer = VariableReplayBuffer(capacity, horizon, 0.1)

    def gen_check(ac_id, i):
        return ac_id * 10000 + i

    def add_step(gs, step):
        dones = (actor_ids * 16 + 16 == step % (8 * 16 + 16)).float()
        obs = actor_ids, torch.full_like(actor_ids, step), dones, gen_check(actor_ids, step), torch.full_like(actor_ids, gs)
        obs = torch.stack([o.float() for o in obs], 1)
        # print(i, dones, obs)
        buffer.push(actor_ids, dones=dones, obs=obs)

    for step in range(horizon):
        add_step(0, step)
    for k, v in buffer.get_new_samples().items():
        assert v.shape[0] == horizon and v.shape[1] == num_actors, (k, v.shape)

    for gs in range(num_gs):
        for step in range(iter_len):
            add_step(gs, step)

    # print()
    # print(len(buffer), len(buffer._buffers), [len(b) for b in buffer._buffers])
    # print(buffer._actor_id_to_index)
    # print()

    assert buffer.sample(64)['dones'].sum().item() > 0
    buffer.get_new_samples()

    for i, v in enumerate(zip(*buffer.sample(64).values())):
        done, obs = v
        actor_ids, step, tdones, check, gs = obs.unbind(1)
        # print(i, actor_ids, step, tdones)
        assert gs.mean().item() > num_gs - 2
        assert torch.allclose(check, gen_check(actor_ids, step))
        assert torch.allclose(done, tdones)


def test_variable_replay_buffer_variable():
    torch.manual_seed(345)
    random.seed(345)

    min_actors = 1
    max_actors = 16
    capacity = 5000
    horizon = 64
    episode_end_chance = 1 / 16

    buffer = VariableReplayBuffer(capacity, horizon, 0.1)

    actor_ids = torch.randperm(10000)[:random.randint(min_actors, max_actors)].tolist()
    local_step = len(actor_ids) * [0]

    def gen_check(ac_id, local_step):
        return ac_id * 10000 + local_step

    for global_step in range(10000):
        n_ac = len(actor_ids)
        t_dones = torch.zeros(n_ac)
        next_actor_ids = actor_ids.copy()
        next_local_step = [s + 1 for s in local_step]

        if random.random() < episode_end_chance and n_ac > min_actors:
            end_index = random.randrange(n_ac)
            t_dones[end_index] = 1
            next_actor_ids.pop(end_index)
            next_local_step.pop(end_index)
        if random.random() < episode_end_chance and n_ac < max_actors:
            next_actor_ids.append(random.randrange(10000))
            next_local_step.append(0)

        t_aid = torch.as_tensor(actor_ids)
        t_gs = torch.full((n_ac,), global_step)
        t_ls = torch.as_tensor(local_step)
        obs = t_aid, t_dones, t_gs, t_ls, gen_check(t_aid, t_ls)
        obs = torch.stack([o.float() for o in obs], 1)
        buffer.push(t_aid, dones=t_dones, obs=obs)

        actor_ids = next_actor_ids
        local_step = next_local_step

    # print()
    # print(len(buffer), len(buffer._buffers), [len(b) for b in buffer._buffers])
    # print(buffer._actor_id_to_index)
    # print()

    dmean = buffer.sample(64)[DONE].mean().item()
    cor_es = episode_end_chance * 2 / (min_actors + max_actors)
    assert cor_es / 1.5 < dmean < cor_es * 1.5

    buffer.get_new_samples()

    for i, v in enumerate(zip(*buffer.sample(8).values())):
        done, obs = v
        actor_ids,  tdones, gs, ls, check = obs.unbind(1)
        # print(i, actor_ids, ls, tdones)
        assert gs.mean().item() > 5000
        assert torch.allclose(check, gen_check(actor_ids, ls))
        assert torch.allclose(done, tdones)
