import dataclasses
import random
from collections import defaultdict
from dataclasses import dataclass
from itertools import count
from typing import NamedTuple, List, Optional, Dict, Tuple, Callable
from unittest.mock import Mock

from .reward_weight_generator import RewardWeightGenerator
from torch import Tensor

import torch
from ..actors import Actor, ModularActor
from .variable_replay_buffer import VariableReplayBuffer
from ..common.attr_dict import AttrDict
from ..common.rl_base import RLStepData


@dataclass
class StepData:
    obs: Tensor = None
    input_memory: Tensor = None
    output_memory: Tensor = None
    values: Tensor = None
    logits: Tensor = None
    actions: Tensor = None
    rewards: Tensor = None
    done: Tensor = None
    reward_weights: Tensor = None


class VariableStepCollector:
    def __init__(self, actor: Actor, replay_buffer: VariableReplayBuffer, device: torch.device,
                 reward_weight_gen: RewardWeightGenerator, reward_reweight_interval: int):
        self.actor = actor
        self.replay_buffer = replay_buffer
        self.device = device
        self.reward_reweight_interval = reward_reweight_interval
        self.reward_weight_gen = reward_weight_gen
        self._step_datas: Dict[int, StepData] = defaultdict(StepData)
        self._memory_shape = None
        self._reward_reweight_counter = 0

    def step(self, data: RLStepData) -> Tensor:
        with torch.no_grad():
            self._reward_reweight_counter += 1
            if self._reward_reweight_counter >= self.reward_reweight_interval:
                self._reward_reweight_counter = 0
                self._reweight_rewards()

            values, logits, reward_weights, memory_in, memory_out = self._run_model(data)

            actions = self.actor.heads.logits.pd.sample(logits)
            assert not torch.isnan(actions.sum())

            self._push_to_buffer(data, values, logits, actions, reward_weights, memory_in, memory_out)
            return actions

    def drop_collected_steps(self):
        self._step_datas.clear()

    def _run_model(self, data: RLStepData) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
        obs = data.obs.to(self.device)
        reward_weights = self._get_data_field(data.actor_id, data.done,
                                              lambda x: x.reward_weights,
                                              lambda: self.reward_weight_gen.generate(1, device=self.device)[0])
        if self.actor.is_recurrent:
            dones_t = data.done.unsqueeze(0).to(self.device)
            if self._memory_shape is None:
                self._memory_shape = self.actor(obs.unsqueeze(0), memory=None, dones=dones_t,
                                                goal=reward_weights.unsqueeze(0)).memory.shape[1:]
            input_memory = self._get_data_field(data.actor_id, data.done,
                                                lambda x: x.output_memory,
                                                lambda: torch.zeros(self._memory_shape))
            ac_out = self.actor(obs.unsqueeze(0), memory=input_memory, dones=dones_t, goal=reward_weights.unsqueeze(0))
            ac_out.logits, ac_out.state_values = [x.squeeze(0) for x in (ac_out.logits, ac_out.state_values)]
        else:
            ac_out = self.actor(obs, goal=reward_weights)

        ac_out = AttrDict({k: v.cpu() for k, v in ac_out.items()})
        ac_out.state_values = ac_out.state_values.squeeze(-1)

        mem_tuple = (input_memory, ac_out.memory) if self.actor.is_recurrent else (None, None)
        return (ac_out.state_values, ac_out.logits, reward_weights, *mem_tuple)

    def _push_to_buffer(self, data: RLStepData, values, logits, actions, reward_weights, memory_in, memory_out):
        # write rewards and done to step data, extract complete steps
        full_step_datas = {}
        for i, (aid, done) in enumerate(zip(data.actor_id.tolist(), data.done.tolist())):
            if aid not in self._step_datas:
                continue
            step_data = self._step_datas[aid]
            step_data.rewards = data.rewards[i]
            step_data.done = data.done[i]
            full_step_datas[aid] = step_data
            if done:
                del self._step_datas[aid]

        if len(full_step_datas) > 0:
            # convert full_step_datas to tensors
            full_actor_ids, full_step_datas = list(full_step_datas.keys()), list(full_step_datas.values())
            full_step_datas = [dataclasses.asdict(x) for x in full_step_datas]
            full_step_datas = {k: torch.stack([d[k] for d in full_step_datas], 0)
                               for k in full_step_datas[0].keys() if full_step_datas[0][k] is not None}
            full_step_datas = StepData(**full_step_datas)

            # push to buffer
            self.replay_buffer.push(
                actor_ids=torch.tensor(full_actor_ids, dtype=torch.long),
                states=full_step_datas.obs,
                state_values=full_step_datas.values,
                logits=full_step_datas.logits,
                actions=full_step_datas.actions,
                **(dict(memory=full_step_datas.input_memory) if self.actor.is_recurrent else {}),
                rewards=full_step_datas.rewards,
                dones=full_step_datas.done,
            )

        # add new steps to _step_datas
        for i, (aid, done) in enumerate(zip(data.actor_id.tolist(), data.done.tolist())):
            if done:
                continue
            step_data = self._step_datas[aid]
            step_data.obs = data.obs[i]
            step_data.values = values[i]
            step_data.logits = logits[i]
            step_data.actions = actions[i]
            step_data.reward_weights = reward_weights[i]
            step_data.rewards = None
            step_data.done = None
            if self.actor.is_recurrent:
                step_data.input_memory = memory_in[i]
                step_data.output_memory = memory_out[i]

    def _get_data_field(self, actor_id: Tensor, done: Tensor,
                        get_field: Callable[[StepData], Tensor], new_fn: Callable[[], Tensor]) -> Tensor:
        fields = []
        for aid, done in zip(actor_id.tolist(), done.tolist()):
            if done or aid not in self._step_datas:
                fields.append(new_fn())
            else:
                fields.append(get_field(self._step_datas[aid]))
        return torch.stack(fields, 0).to(self.device)

    def _reweight_rewards(self):
        reward_weights = self.reward_weight_gen.generate(len(self._step_datas), self.device)
        for (aid, data), rw in zip(self._step_datas.items(), reward_weights):
            data.reward_weights = rw


def test_variable_step_collector_reweight():
    torch.manual_seed(123)
    random.seed(123)

    num_actors = 2
    capacity = 8 * num_actors
    horizon = 8
    actor_ids = torch.arange(num_actors)

    actor = Mock()
    actor.is_recurrent = False
    actor.return_value = AttrDict(logits=torch.zeros(num_actors, 2), state_values=torch.zeros(num_actors, 1))
    actor.heads.logits.pd.sample.return_value = torch.zeros(num_actors, 2)
    buffer = VariableReplayBuffer(capacity, horizon, 0.1)
    rw_gen = RewardWeightGenerator(3)
    collector = VariableStepCollector(actor, buffer, torch.device('cpu'), rw_gen, 5)

    datas = []
    def step():
        data = RLStepData(rewards=torch.rand((num_actors, 1)), true_reward=torch.rand(num_actors),
                          done=torch.zeros(num_actors), obs=torch.rand((num_actors, 1)), actor_id=actor_ids)
        datas.append(data)
        collector.step(data)

    step()
    weights = [d.reward_weights for d in collector._step_datas.values()]
    assert all(w is not None for w in weights)
    step()
    weights2 = [d.reward_weights for d in collector._step_datas.values()]
    assert all(torch.allclose(a, b) for a, b in zip(weights, weights2))
    # step()
    # step()
    # step()
    # weights2 = [d.reward_weights for d in collector._step_datas.values()]
    # assert not all(torch.allclose(a, b) for a, b in zip(weights, weights2))
    # print()
    # for d in datas:
    #     print(d)
    # print()
    # for b in buffer._buffers:
    #     print(len(b))
    #     for k, v in b._data.items():
    #         print(k, v)


def test_variable_step_collector_normal():
    torch.manual_seed(123)
    random.seed(123)

    num_actors = 8
    capacity = 512 * num_actors
    horizon = 64
    num_gs = 10
    iter_len = 512
    actor_ids = torch.arange(num_actors)

    actor = Mock()
    actor.is_recurrent = False
    actor.return_value = AttrDict(logits=torch.zeros(num_actors, 2), state_values=torch.zeros(num_actors, 1))
    actor.heads.logits.pd.sample.return_value = torch.zeros(num_actors, 2)
    buffer = VariableReplayBuffer(capacity, horizon, 0.1)
    rw_gen = RewardWeightGenerator(4)
    collector = VariableStepCollector(actor, buffer, torch.device('cpu'), rw_gen, 5)

    def gen_check(ac_id, i):
        return ac_id * 10000 + i

    def add_step(gs, step):
        dones = (actor_ids * 16 + 16 == step % (8 * 16 + 16)).float()
        dones_next = (actor_ids * 16 + 16 == (step + 1) % (8 * 16 + 16)).float()
        obs = actor_ids, torch.full_like(actor_ids, step), dones_next, gen_check(actor_ids, step), torch.full_like(actor_ids, gs)
        obs = torch.stack([o.float() for o in obs], 1)
        # print(i, dones, obs)
        collector.step(RLStepData(rewards=torch.zeros((num_actors, 1)), true_reward=torch.zeros(num_actors),
                                  done=dones, obs=obs, actor_id=actor_ids))

    for step in range(horizon + 4):  # +4 because episode end could reduce it
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

    samples = AttrDict(buffer.sample(64))
    for i in range(horizon):
        states, dones = samples.states[i], samples.dones[i]
        actor_ids, step, tdones, check, gs = states.unbind(1)
        # print(i, actor_ids, step, tdones)
        assert gs.mean().item() > num_gs - 2
        assert torch.allclose(check, gen_check(actor_ids, step))
        assert torch.allclose(dones, tdones)


def test_variable_step_collector_normal_recurrent():
    torch.manual_seed(123)
    random.seed(123)

    num_actors = 8
    capacity = 512 * num_actors
    horizon = 64
    num_gs = 10
    iter_len = 512
    actor_ids = torch.arange(num_actors)

    actor = Mock()
    actor.is_recurrent = True
    actor.side_effect = lambda *args, **kwargs: AttrDict(
        logits=torch.zeros(num_actors, 2),
        state_values=torch.zeros(num_actors, 1),
        memory=torch.tensor(num_actors * [actor.mem_data], dtype=torch.float)
    )
    actor.heads.logits.pd.sample = Mock(return_value=torch.zeros(num_actors, 2))
    buffer = VariableReplayBuffer(capacity, horizon, 0.1)
    rw_gen = RewardWeightGenerator(4)
    collector = VariableStepCollector(actor, buffer, torch.device('cpu'), rw_gen, 1)

    def gen_check(ac_id, i):
        return ac_id * 10000 + i

    def add_step(gs, step):
        actor.mem_data = (gs, step)
        dones = (actor_ids * 16 + 16 == step % (8 * 16 + 16)).float()
        dones_next = (actor_ids * 16 + 16 == (step + 1) % (8 * 16 + 16)).float()
        obs = actor_ids, torch.full_like(actor_ids, step), dones_next, gen_check(actor_ids, step), torch.full_like(actor_ids, gs)
        obs = torch.stack([o.float() for o in obs], 1)
        # print(i, dones, obs)
        collector.step(RLStepData(rewards=torch.zeros((num_actors, 1)), true_reward=torch.zeros(num_actors),
                                  done=dones, obs=obs, actor_id=actor_ids))

    for step in range(horizon + 4):  # +4 because episode end could reduce it
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

    samples = AttrDict(buffer.sample(64))
    for i in range(horizon):
        states, dones = samples.states[i], samples.dones[i]
        actor_ids, step, tdones, check, gs = states.unbind(1)
        # print(i, actor_ids, step, tdones)
        assert gs.mean().item() > num_gs - 2
        assert torch.allclose(check, gen_check(actor_ids, step))
        assert torch.allclose(dones, tdones)


# def test_variable_step_collector_variable():
#     torch.manual_seed(345)
#     random.seed(345)
#
#     min_actors = 1
#     max_actors = 16
#     capacity = 5000
#     horizon = 64
#     episode_end_chance = 1 / 16
#
#     buffer = VariableReplayBuffer(capacity, horizon, 0.1)
#
#     actor_ids = torch.randperm(10000)[:random.randint(min_actors, max_actors)].tolist()
#     local_step = len(actor_ids) * [0]
#
#     def gen_check(ac_id, local_step):
#         return ac_id * 10000 + local_step
#
#     for global_step in range(10000):
#         n_ac = len(actor_ids)
#         t_dones = torch.zeros(n_ac)
#         next_actor_ids = actor_ids.copy()
#         next_local_step = [s + 1 for s in local_step]
#
#         if random.random() < episode_end_chance and n_ac > min_actors:
#             end_index = random.randrange(n_ac)
#             t_dones[end_index] = 1
#             next_actor_ids.pop(end_index)
#             next_local_step.pop(end_index)
#         if random.random() < episode_end_chance and n_ac < max_actors:
#             next_actor_ids.append(random.randrange(10000))
#             next_local_step.append(0)
#
#         t_aid = torch.as_tensor(actor_ids)
#         t_gs = torch.full((n_ac,), global_step)
#         t_ls = torch.as_tensor(local_step)
#         obs = t_aid, t_dones, t_gs, t_ls, gen_check(t_aid, t_ls)
#         obs = torch.stack([o.float() for o in obs], 1)
#         buffer.push(t_aid, dones=t_dones, obs=obs)
#
#         actor_ids = next_actor_ids
#         local_step = next_local_step
#
#     # print()
#     # print(len(buffer), len(buffer._buffers), [len(b) for b in buffer._buffers])
#     # print(buffer._actor_id_to_index)
#     # print()
#
#     dmean = buffer.sample(64)[DONE].mean().item()
#     cor_es = episode_end_chance * 2 / (min_actors + max_actors)
#     assert cor_es / 1.5 < dmean < cor_es * 1.5
#
#     buffer.get_new_samples()
#
#     for i, v in enumerate(zip(*buffer.sample(8).values())):
#         done, obs = v
#         actor_ids,  tdones, gs, ls, check = obs.unbind(1)
#         # print(i, actor_ids, ls, tdones)
#         assert gs.mean().item() > 5000
#         assert torch.allclose(check, gen_check(actor_ids, ls))
#         assert torch.allclose(done, tdones)