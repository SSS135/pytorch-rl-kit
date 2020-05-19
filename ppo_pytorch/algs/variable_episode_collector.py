from collections import defaultdict
from itertools import count
from typing import NamedTuple, List, Optional, Dict

import torch
from ppo_pytorch.actors.actors import Actor
from ppo_pytorch.common.attr_dict import AttrDict
from ppo_pytorch.common.rl_base import RLStepData


class EpisodeData(NamedTuple):
    obs: List[torch.Tensor]
    rewards: List[torch.Tensor]
    value: List[torch.Tensor]
    memory: Optional[List[torch.Tensor]]
    logits: List[torch.Tensor]
    actions: List[torch.Tensor]


class CollectorOut(NamedTuple):
    actions: torch.Tensor
    episodes: List[EpisodeData]


class VariableEpisodeCollector:
    def __init__(self, actor: Actor, device: torch.device):
        self.actor = actor
        self.device = device
        self._episodes: Dict[int, EpisodeData] = defaultdict(lambda: EpisodeData([], [], [], [], [], []))
        self._memory_shape = None

    def step(self, data: RLStepData) -> CollectorOut:
        with torch.no_grad():
            values, input_memory, memory, logits = self._run_model(data)

            actions = self.actor.heads.logits.pd.sample(logits)
            assert not torch.isnan(actions.sum())

            self._add_to_buffer(data, input_memory, memory, values, logits, actions)
            finished_episodes = self._extract_finished_episodes(data)
            return CollectorOut(actions, finished_episodes)

    def drop_collected_steps(self):
        self._episodes.clear()

    def _run_model(self, data: RLStepData):
        obs = data.obs.to(self.device)
        if self.actor.is_recurrent:
            dones_t = data.done.unsqueeze(0).to(self.device)
            if self._memory_shape is None:
                self._memory_shape = self.actor(obs.unsqueeze(0), memory=None, dones=dones_t).memory.shape[1:]
            input_memory = self._get_memory(data.actor_id)
            ac_out = self.actor(obs.unsqueeze(0), memory=input_memory, dones=dones_t)
            ac_out.logits, ac_out.state_values = [x.squeeze(0) for x in (ac_out.logits, ac_out.state_values)]
        else:
            ac_out = self.actor(obs)
            ac_out.memory = input_memory = None

        ac_out = AttrDict({k: v.cpu() for k, v in ac_out.items()})
        ac_out.state_values = ac_out.state_values.squeeze(-1)
        return ac_out.state_values, input_memory, ac_out.memory, ac_out.logits

    def _add_to_buffer(self, data: RLStepData, input_memory, memory, values, logits, actions):
        mem_tuple = (input_memory, memory) if self.actor.is_recurrent else (count(), count())
        ep_tuple = (data.actor_id, data.obs, data.rewards, values, logits, actions, *mem_tuple)
        for aid, obs, r, V, lg, ac, m_in, m_out in zip(*ep_tuple):
            aid = aid.item()
            ep = self._episodes[aid]
            ep.obs.append(obs)
            ep.rewards.append(r)
            ep.value.append(V)
            ep.logits.append(lg)
            ep.actions.append(ac)
            if self.actor.is_recurrent:
                if len(ep.memory) == 0:
                    ep.memory.append(m_in)
                ep.memory.append(m_out)

    def _extract_finished_episodes(self, data: RLStepData) -> List[EpisodeData]:
        ep_data = []
        for aid, done in zip(data.actor_id, data.done):
            aid, done = aid.item(), done.item()
            if not done or aid not in self._episodes:
                continue
            ep = self._episodes[aid]
            del self._episodes[aid]
            ep.rewards.pop(0)
            ep.obs.pop()
            ep.value.pop()
            ep.actions.pop()
            ep.logits.pop()
            if self.actor.is_recurrent:
                ep.memory.pop()
                ep.memory.pop()
            else:
                ep.memory = None
            ep_data.append(ep)
        return ep_data

    def _get_memory(self, actor_id: torch.Tensor) -> torch.Tensor:
        mem = []
        for aid in actor_id:
            aid = aid.item()
            if aid in self._episodes:
                mem.append(self._episodes[aid].memory[-1])
            else:
                mem.append(torch.zeros(self._memory_shape, device=self.device))
        return torch.stack(mem, 0)