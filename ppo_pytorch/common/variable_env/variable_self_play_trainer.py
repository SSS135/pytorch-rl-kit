import pprint
import random
from collections import OrderedDict
from collections import deque
from dataclasses import dataclass
from typing import Callable, Tuple, Optional, Dict, Set, Deque, List

import numpy as np
import torch
from torch import Tensor
from trueskill import Rating, rate_1vs1

from .variable_env import VariableVecEnv
from ..rl_base import RLBase
from ..tensorboard_env_logger import TensorboardEnvLogger


@dataclass
class Team:
    id: int
    selfplay: List[bool]
    done: List[bool]
    returns: List[float]
    agents: List[int]


@dataclass
class Match:
    id: int
    teams: Dict[int, Team]

    @property
    def done(self):
        return all(d for t in self.teams.values() for d in t.done)


@dataclass
class ArchiveEntry:
    rating: Rating
    state_dict: OrderedDict


class VariableSelfPlayTrainer:
    def __init__(self,
                 rl_alg_factory: Callable[..., RLBase],
                 env_factory: Callable[[], VariableVecEnv],
                 num_archive_models: int,
                 archive_save_interval: int,
                 archive_switch_interval: int,
                 selfplay_prob: float,
                 log_root_path: str,
                 log_interval: int = 10000,
                 alg_name: str = 'RL',
                 tag: str = ''):
        self._init_args = locals()
        self.rl_alg_factory = rl_alg_factory
        self.num_archive_models = num_archive_models
        self.archive_save_interval = archive_save_interval
        self.archive_switch_interval = archive_switch_interval
        self.selfplay_prob = selfplay_prob
        self.log_interval = log_interval
        self.alg_name = alg_name
        self.tag = tag

        assert 0 <= selfplay_prob <= 1
        assert log_root_path is not None

        self._env = env_factory()
        log_dir = TensorboardEnvLogger.get_log_dir(log_root_path, alg_name, self._env.env_name, tag)
        print('Log dir:', log_dir)
        self._data = self._env.reset()

        self._rl_alg = rl_alg_factory(
            self._env.observation_space, self._env.action_space, num_rewards=len(self._data.reward_names),
            log_interval=log_interval, model_save_folder=log_dir)
        self._archive_rl_alg = rl_alg_factory(
            self._env.observation_space, self._env.action_space, num_rewards=len(self._data.reward_names),
            disable_training=True)

        self.logger = TensorboardEnvLogger(alg_name, self._env.env_name, log_dir, log_interval, tag=tag)
        self.logger.add_text('EnvTrainer', pprint.pformat(self._init_args))
        self._rl_alg.logger = self.logger

        self._frame = 0
        self._archive: Deque[ArchiveEntry] = deque(maxlen=num_archive_models)
        self._current_model: Optional[ArchiveEntry] = None
        self._last_save_frame = 0
        self._last_switch_frame = 0
        self._matches: Dict[int, Match] = {}
        self._matches_wr = []
        self._main_rating = Rating(25, 1)
        self._cpu = torch.device('cpu')
        self._last_log_frame = 0
        self._actors_sp = set()
        self._actors_arch = set()

    def step(self):
        self._save_archive_model()
        self._select_archive_model()

        tensors_sp, tensors_arch = self._separate_data()
        ac_sp = tensors_sp[4] if tensors_sp is not None else None
        ac_arch = tensors_arch[4] if tensors_arch is not None else None
        self._validate_actors(ac_sp, ac_arch)
        action_sp = self._rl_alg.step(*tensors_sp) if tensors_sp is not None else None
        action_arch = self._archive_rl_alg.step(*tensors_arch) if tensors_arch is not None else None
        action = self._cat_actions(action_sp, action_arch)
        assert torch.allclose(torch.from_numpy(self._data.agent_id), self._cat_actions(ac_sp, ac_arch))

        self._update_returns()
        self._set_done_flags()
        self._rate_matches()
        self._cleanup_matches()
        self._data = self._env.step(action.numpy())

        self._frame += len(self._data.obs)

        if self.logger is not None:
            self.logger.step(self._data, False)
            self._log_rating()

    def train(self, max_frames):
        while self._frame < max_frames:
            self.step()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._env.close()

    def _validate_actors(self, actors_sp: Optional[Tensor], actors_arch: Optional[Tensor]):
        actors_sp = [] if actors_sp is None else actors_sp.tolist()
        actors_arch = [] if actors_arch is None else actors_arch.tolist()
        for a in actors_sp:
            self._actors_sp.add(a)
        for a in actors_arch:
            self._actors_arch.add(a)
        if len(self._actors_sp) > 10000:
            self._actors_sp.clear()
            self._actors_arch.clear()
        assert len(self._actors_arch & self._actors_sp) == 0, self._actors_arch & self._actors_sp

    def _separate_data(self):
        data_sp, data_arch = [], []
        for i, (m_id, t_id, a_id) in enumerate(zip(self._data.match_id.tolist(), self._data.team_id.tolist(), self._data.agent_id.tolist())):
            match, team = self._get_match_and_team(m_id, t_id, a_id, allow_create=True)
            sample = [x[i] for x in (self._data.obs, self._data.rewards, self._data.done, self._data.true_reward, self._data.agent_id)]
            (data_sp if team.selfplay[team.agents.index(a_id)] else data_arch).append(sample)
        return [[torch.from_numpy(np.stack(x, 0)) for x in zip(*data)] if len(data) > 0 else None for data in (data_sp, data_arch)]

    def _cat_actions(self, actions_sp: Optional[Tensor], actions_arch: Optional[Tensor]) -> Tensor:
        actions_sp, actions_arch = [list(x) if x is not None else [] for x in (actions_sp, actions_arch)]
        actions = []
        for m_id, t_id, a_id in zip(self._data.match_id.tolist(), self._data.team_id.tolist(), self._data.agent_id.tolist()):
            match, team = self._get_match_and_team(m_id, t_id, a_id)
            actions.append((actions_sp if team.selfplay[team.agents.index(a_id)] else actions_arch).pop(0))
        return torch.stack(actions, 0)

    def _get_match_and_team(self, m_id: int, t_id: int, a_id: int, allow_create=False) -> Tuple[Match, Team]:
        match = self._matches.get(m_id)
        if match is None:
            assert allow_create
            match = self._matches[m_id] = Match(m_id, {})
            # print('create match', m_id)

        team = match.teams.get(t_id)
        if team is None:
            assert allow_create
            team = match.teams[t_id] = Team(t_id, [], [], [], [])
            # print('create team', t_id, 'match', m_id)

        if a_id not in team.agents:
            team.agents.append(a_id)
            team.selfplay.append(random.random() < self.selfplay_prob)
            team.done.append(False)
            team.returns.append(0)
            assert all(t == team or a_id not in t.agents for m in self._matches.values() for t in m.teams.values()), \
                (m_id, t_id, a_id, self._matches)
            # print('create agent', a_id, 'team', t_id, 'match', m_id)

        return match, team

    def _rate_matches(self):
        for m_id in self._data.match_id.tolist():
            if m_id not in self._matches:
                continue
            teams = self._matches[m_id].teams.values()
            if not all(done for team in teams for done in team.done):
                continue

            selfplay_all = all(s for t in teams for s in t.selfplay)
            archive_all = all(not s for t in teams for s in t.selfplay)
            if selfplay_all or archive_all:
                continue

            r_selfplay = np.mean([r for t in teams for (r, selfplay) in zip(t.returns, t.selfplay) if selfplay])
            r_archive = np.mean([r for t in teams for (r, selfplay) in zip(t.returns, t.selfplay) if not selfplay])
            win = r_selfplay > r_archive
            draw = r_selfplay == r_archive

            self._matches_wr.append(float(0.5 if draw else win))
            if win:
                self._main_rating, self._current_model.rating = rate_1vs1(self._main_rating, self._current_model.rating, drawn=draw)
            else:
                self._current_model.rating, self._main_rating = rate_1vs1(self._current_model.rating, self._main_rating, drawn=draw)

    def _log_rating(self):
        if self._last_log_frame + self.log_interval > self._frame:
            return
        self._last_log_frame = self._frame
        self.logger.add_scalar('Rating/TrueSkill Mu', self._main_rating.mu, self._frame)
        self.logger.add_scalar('Rating/TrueSkill Mu3Sigma', self._main_rating.mu - 3 * self._main_rating.sigma, self._frame)
        if len(self._matches_wr) > 0:
            self.logger.add_scalar('Rating/Win Rate', np.mean(self._matches_wr), self._frame)
            self._matches_wr.clear()

    def _set_done_flags(self):
        for_var = enumerate(zip(self._data.match_id.tolist(), self._data.team_id.tolist(), self._data.agent_id.tolist(), self._data.done.tolist()))
        for i, (m_id, t_id, a_id, done) in for_var:
            if not done:
                continue
            match, team = self._get_match_and_team(m_id, t_id, a_id)
            team.done[team.agents.index(a_id)] = True

    def _update_returns(self):
        for_var = zip(self._data.match_id.tolist(), self._data.team_id.tolist(), self._data.agent_id.tolist(), self._data.true_reward.tolist())
        for m_id, t_id, a_id, r in for_var:
            match, team = self._get_match_and_team(m_id, t_id, a_id)
            team.returns[team.agents.index(a_id)] += r

    def _cleanup_matches(self):
        for m_id in self._data.match_id.tolist():
            if m_id in self._matches and all(d for t in self._matches[m_id].teams.values() for d in t.done):
                del self._matches[m_id]
                # print('del match', m_id)

    def _save_archive_model(self):
        if len(self._archive) == 0 or self._last_save_frame + self.archive_save_interval < self._frame:
            self._last_save_frame = self._frame
            state_dict = self._rl_alg._eval_model.state_dict()
            state_dict = OrderedDict([(k, v.to(self._cpu, copy=True)) for k, v in state_dict.items()])
            self._archive.append(ArchiveEntry(self._main_rating, state_dict))

    def _select_archive_model(self):
        if self._current_model is None or self._last_switch_frame + self.archive_switch_interval < self._frame:
            self._last_switch_frame = self._frame
            self._current_model = random.choice(self._archive)
            self._archive_rl_alg._eval_model.load_state_dict(self._current_model.state_dict)