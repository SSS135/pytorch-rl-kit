from asyncio import Future
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from functools import partial
from typing import Optional

import gym.spaces
import torch
import torch.autograd
import torch.autograd
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torchvision.utils import make_grid

from .replay_buffer import ReplayBuffer
from .utils import blend_models
from ..actors import ModularActor, create_td3_fc_actor
from ..actors.utils import model_diff
from ..common.attr_dict import AttrDict
from ..common.barron_loss import barron_loss
from ..common.data_loader import DataLoader
from ..common.probability_distributions import DiagGaussianPd, CategoricalPd, make_pd
from ..common.rl_base import RLBase
import torch.nn as nn
from ..common.gae import calc_value_targets


N_H = 128


class MuZeroPredictionNet(nn.Module):
    def __init__(self, obs_len, action_len, logits_len):
        super().__init__()
        self.obs_len = obs_len
        self.action_len = action_len
        self.logits_len = logits_len
        self.start_norm = nn.LayerNorm(N_H)
        self.action_encoder = nn.Linear(action_len, N_H)
        self.gate = nn.Sequential(nn.Linear(N_H, N_H), nn.Sigmoid())
        self.reward_value_logits_encoder = nn.Linear(N_H, 2 + logits_len)
        self.next_state_model = nn.Sequential(
            # nn.LayerNorm(N_H),
            nn.LeakyReLU(0.1),
            nn.Linear(N_H, N_H),
            nn.LeakyReLU(0.1),
            nn.Linear(N_H, N_H),
        )
        self.observation_encoder = nn.Sequential(
            nn.Linear(obs_len, N_H),
            nn.LeakyReLU(0.1),
            nn.Linear(N_H, N_H),
            # nn.LayerNorm(N_H),
        )

    def forward(self, hidden, actions):
        x = self.start_norm(hidden) + self.action_encoder(actions)
        x = self.next_state_model(x)
        hidden = hidden + self.gate(x) * x
        return self._get_output(hidden)

    def encode_observation(self, observation):
        hidden = self.observation_encoder(observation)
        return self._get_output(hidden)

    def _get_output(self, hidden):
        reward, value, logits = self.reward_value_logits_encoder(hidden).split_with_sizes([1, 1, self.logits_len], -1)
        return AttrDict(hidden=hidden, rewards=reward.squeeze(-1), state_values=value.squeeze(-1), logits=logits)


class MuZero(RLBase):
    def __init__(self, observation_space, action_space,
                 reward_discount=0.99,
                 horizon=16,
                 batch_size=32,
                 num_batches=4,
                 num_actions=4,
                 num_rollouts=16,
                 rollout_depth=5,
                 replay_buffer_size=128*1024,
                 target_model_blend=0.005,
                 optimizer_factory=partial(optim.Adam, lr=3e-4),
                 cuda_eval=True,
                 cuda_train=True,
                 grad_clip_norm=None,
                 reward_scale=1.0,
                 barron_alpha_c=(1.5, 1),
                 lr_scheduler_factory=None,
                 entropy_decay_factory=None,
                 **kwargs):
        super().__init__(observation_space, action_space, **kwargs)
        self._init_args = locals()
        self.reward_discount = reward_discount
        self.horizon = horizon
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.replay_buffer_size = replay_buffer_size
        self.target_model_blend = target_model_blend
        self.device_eval = torch.device('cuda' if cuda_eval else 'cpu')
        self.device_train = torch.device('cuda' if cuda_train else 'cpu')
        self.grad_clip_norm = grad_clip_norm
        self.reward_scale = reward_scale
        self.barron_alpha_c = barron_alpha_c
        self.num_rollouts = num_rollouts
        self.rollout_depth = rollout_depth
        self.num_actions = num_actions

        self.pd = make_pd(action_space)
        self._train_model: MuZeroPredictionNet = MuZeroPredictionNet(
            observation_space.shape[0], self.pd.input_vector_len, self.pd.prob_vector_len)

        if self.model_init_path is not None:
            self._train_model.load_state_dict(torch.load(self.model_init_path))
            print(f'loaded model {self.model_init_path}')

        self._train_model = self._train_model.to(self.device_train)
        self._eval_model = deepcopy(self._train_model).to(self.device_eval)

        self._optimizer: torch.optim.Optimizer = optimizer_factory(self._train_model.parameters())
        self._replay_buffer = ReplayBuffer(replay_buffer_size)
        # self._pop_art = PopArt()
        # self._train_executor = ThreadPoolExecutor(max_workers=1)
        self._eval_steps = 0
        self._prev_data = None
        # self._train_future: Optional[Future] = None
        # self._update_iter = 0

    def _step(self, rewards, dones, states) -> torch.Tensor:
        with torch.no_grad():
            logits, actions, state_values = self._search_action(states.to(self.device_eval))
            actions = actions.cpu()

            if not self.disable_training:
                if self._prev_data is not None and rewards is not None:
                    self._replay_buffer.push(rewards=rewards, dones=dones, **self._prev_data)

                self._eval_steps += 1
                self._prev_data = dict(logits=logits, states=states, actions=actions, state_values=state_values)

                if self._eval_steps > self.horizon and len(self._replay_buffer) >= self.batch_size:
                    self._eval_steps = 0
                    self._pre_train()
                    self._train()

            return actions

    def _pre_train(self):
        self._check_log()

        # # update clipping and learning rate decay schedulers
        # if self._actor_lr_scheduler is not None:
        #     self._actor_lr_scheduler.step(self.frame)
        # if self._critic_lr_scheduler is not None:
        #     self._critic_lr_scheduler.step(self.frame)
        # if self._entropy_decay is not None:
        #     self._entropy_decay.step(self.frame)

    def _train(self):
        data = self._create_data()
        self._do_train(data)
        # if self._train_future is not None:
        #     self._train_future.result()
        # self._train_future = self._train_executor.submit(self._train_async, data)

    def _do_train(self, data):
        with torch.no_grad():
            self._log_training_data(data)
            self._muzero_update(data)
            self._model_saver.check_save_model(self._train_model, self.frame)

    def _create_data(self):
        # (steps, actors, *)
        data = self._replay_buffer.get_last_samples(self.horizon)
        data = AttrDict(data)
        data.rewards = self.reward_scale * data.rewards
        return data

    def _muzero_update(self, data: AttrDict):
        num_samples = data.states.shape[1]

        state_values_p1 = torch.cat([data.state_values, self._prev_data['state_values'].unsqueeze(0)], 0)
        data.value_targets = calc_value_targets(data.rewards, state_values_p1, data.dones, self.reward_discount, 0.99)

        rand_idx = torch.randperm(num_samples, device=self.device_train).chunk(self.num_batches)

        old_model = deepcopy(self._train_model)

        with DataLoader(data, rand_idx, self.device_train, 4, dim=1) as data_loader:
            for batch_index in range(self.num_batches):
                # prepare batch data
                batch = AttrDict(data_loader.get_next_batch())
                loss = self._update_traj(batch, self._do_log and batch_index == self.num_batches - 1)
                # if self._do_actor_update:
                #     blend_models(self._train_model, self._target_model, self.target_model_blend)
                # self._update_iter += 1

        if self._do_log:
            self.logger.add_scalar('learning rate', self._optimizer.param_groups[0]['lr'], self.frame)
            self.logger.add_scalar('total loss', loss, self.frame)
            self.logger.add_scalar('model abs diff', model_diff(old_model, self._train_model), self.frame)
            self.logger.add_scalar('model max diff', model_diff(old_model, self._train_model, True), self.frame)

        self._eval_model = deepcopy(self._train_model).to(self.device_eval).eval()

    # @property
    # def _do_actor_update(self):
    #     return self._update_iter % self.critic_iters == 0

    def _update_traj(self, batch, do_log=False):
        assert batch.value_targets.ndim == 2, batch.value_targets.shape
        assert batch.value_targets.shape == batch.logits.shape[:-1] == \
               batch.dones.shape == batch.rewards.shape == batch.actions.shape[:-1]

        with torch.enable_grad():
            ac_out = self._train_model.encode_observation(batch.states[0])
            hidden = ac_out.hidden

            nonterminals = 1 - batch.dones
            logits = batch.logits[0]
            cum_nonterm = torch.ones_like(batch.dones[0])
            losses = 0
            for i in range(batch.states.shape[0]):
                vtarg = (batch.value_targets[i] * cum_nonterm).detach()
                logits = torch.lerp(logits, batch.logits[i], cum_nonterm).detach()
                losses += F.mse_loss(ac_out.state_values, vtarg)
                losses += F.mse_loss(ac_out.logits, logits) - self.pd.logp(batch.actions[i], logits).mean()

                input_actions = self.pd.to_inputs(batch.actions[i])
                ac_out = self._train_model(hidden, input_actions)
                rtarg = (batch.rewards[i] * cum_nonterm).detach()
                cum_nonterm *= nonterminals[i]
                losses += F.mse_loss(ac_out.rewards, rtarg)

            loss = losses.div(batch.states.shape[0])

        if do_log:
            self.logger.add_scalar('reward rmse', (ac_out.rewards - rtarg).pow(2).mean().sqrt(), self.frame)
            self.logger.add_scalar('state_values rmse', (ac_out.state_values - vtarg).pow(2).mean().sqrt(), self.frame)
            self.logger.add_scalar('logits rmse', (ac_out.logits - logits).pow(2).mean().sqrt(), self.frame)

        loss.backward()
        self._optimizer.step()
        self._optimizer.zero_grad()

        return loss

    def _search_action(self, states):
        start_ac_out = self._train_model.encode_observation(states)
        # (B, A, R, X)
        hidden = start_ac_out.hidden\
            .unsqueeze(-2).repeat_interleave(self.num_actions, -2)\
            .unsqueeze(-2).repeat_interleave(self.num_rollouts, -2)
        assert hidden.shape == (*start_ac_out.hidden.shape[:-1], self.num_actions, self.num_rollouts, start_ac_out.hidden.shape[-1])
        actions = self.pd.sample(start_ac_out.logits.unsqueeze(-2).repeat_interleave(self.num_actions, -2))
        start_actions = actions
        actions = actions.unsqueeze(-2).repeat_interleave(self.num_rollouts, -2)
        assert actions.shape[:-1] == hidden.shape[:-1]

        # (B, A, R)
        value_targets = 0
        for i in range(self.rollout_depth):
            input_actions = self.pd.to_inputs(actions)
            ac_out = self._train_model(hidden, input_actions)
            hidden = ac_out.hidden
            actions = self.pd.sample(ac_out.logits)
            value_targets = value_targets + self.reward_discount ** i * ac_out.rewards

        value_targets += self.reward_discount ** (self.rollout_depth - 1) * ac_out.state_values
        value_targets = value_targets.topk(self.num_rollouts // 4, -1)[0]
        # (B, A)
        value_targets = value_targets.mean(-1)
        assert value_targets.shape == hidden.shape[:2]
        ac_idx = value_targets.argmax(-1, keepdim=True)
        actions = start_actions.gather(-2, ac_idx.unsqueeze(-1)).squeeze(-2)
        assert actions.shape[:-1] == hidden.shape[:1]
        return start_ac_out.logits, actions, start_ac_out.state_values

    def drop_collected_steps(self):
        self._prev_data = None

    def _log_training_data(self, data: AttrDict):
        if self._do_log:
            if data.states.dim() == 4:
                if data.states.shape[1] in (1, 3):
                    img = data.states[:4]
                    nrow = 2
                else:
                    img = data.states[:4]
                    img = img.view(-1, *img.shape[2:]).unsqueeze(1)
                    nrow = data.states.shape[1]
                if data.states.dtype == torch.uint8:
                    img = img.float() / 255
                img = make_grid(img, nrow=nrow, normalize=False)
                self.logger.add_image('state', img, self.frame)
            # vsize = data.value_targets.shape[-2] ** 0.5
            # targets = data.value_targets.sum(-2) / vsize
            # values = data.state_values.sum(-2) / vsize
            # v_mean = values.mean(-1)
            # t_mean = targets.mean(-1)
            self.logger.add_scalar('entropy', self.pd.entropy(data.logits).mean(), self.frame)
            self.logger.add_histogram('rewards', data.rewards, self.frame)
            # self.logger.add_histogram('value_targets', targets, self.frame)
            # self.logger.add_histogram('advantages', data.advantages, self.frame)
            self.logger.add_histogram('values', data.state_values, self.frame)
            # self.logger.add_scalar('value rmse', (v_mean - t_mean).pow(2).mean().sqrt(), self.frame)
            # self.logger.add_scalar('value abs err', (v_mean - t_mean).abs().mean(), self.frame)
            # self.logger.add_scalar('value max err', (v_mean - t_mean).abs().max(), self.frame)
            if isinstance(self.pd, DiagGaussianPd):
                mean, std = data.logits.chunk(2, dim=1)
                self.logger.add_histogram('logits mean', mean, self.frame)
                self.logger.add_histogram('logits std', std, self.frame)
            elif isinstance(self.pd, CategoricalPd):
                self.logger.add_histogram('logits log_softmax', F.log_softmax(data.logits, dim=-1), self.frame)
            self.logger.add_histogram('logits', data.logits, self.frame)
            for name, param in self._train_model.named_parameters():
                self.logger.add_histogram(name, param, self.frame)