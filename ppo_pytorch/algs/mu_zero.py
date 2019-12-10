from functools import partial

import torch
import torch.autograd
import torch.autograd
import torch.nn.functional as F
import torch.optim as optim
from ppo_pytorch.actors.cat_value_encoder import CategoricalValueEncoder
from torchvision.utils import make_grid

from .replay_buffer import ReplayBuffer
from ..actors.utils import model_diff, normalized_columns_initializer_
from ..common.attr_dict import AttrDict
from ..common.probability_distributions import DiagGaussianPd, CategoricalPd, make_pd
from ..common.rl_base import RLBase
import torch.nn as nn
from ..common.gae import calc_value_targets
import torch.jit


N_H = 256


# @torch.jit.script
# def min_max_norm(x):
#     min, max = x.min(-1, keepdim=True)[0], x.max(-1, keepdim=True)[0]
#     return (x - min) / (max - min)
#
#
# class MinMaxNorm(nn.Module):
#     def forward(self, x):
#         return min_max_norm(x)


class MuZeroPredictionNet(nn.Module):
    def __init__(self, obs_len: int, action_len: int, logits_len: int,
                 reward_bins: int = 15, value_bins: int = 25, reward_max: float = 15, value_max: float = 300):
        super().__init__()
        self.obs_len = obs_len
        self.action_len = action_len
        self.logits_len = logits_len
        self.reward_bins = reward_bins
        self.value_bins = value_bins
        self.reward_max = reward_max
        self.value_max = value_max
        self.reward_encoder = CategoricalValueEncoder(reward_max, reward_bins)
        self.value_encoder = CategoricalValueEncoder(value_max, value_bins)
        # self.start_norm = nn.LayerNorm(N_H)
        self.action_encoder = nn.Linear(action_len, N_H)
        # self.gate = nn.Sequential(nn.LayerNorm(N_H), nn.LeakyReLU(0.1), nn.Linear(N_H, N_H), nn.Sigmoid())
        self.reward_output = nn.Sequential(
            nn.Linear(N_H, N_H),
            nn.ReLU(),
            nn.Linear(N_H, reward_bins)
        )
        self.value_logits_output = nn.Sequential(
            nn.Linear(N_H, N_H),
            nn.ReLU(),
            nn.Linear(N_H, value_bins + logits_len)
        )
        self.next_state_model = nn.Sequential(
            nn.ReLU(),
            nn.Linear(N_H, N_H),
            nn.ReLU(),
            nn.Linear(N_H, N_H * 2),
        )
        self.observation_model = nn.Sequential(
            nn.Linear(obs_len, N_H),
            nn.ReLU(),
            nn.Linear(N_H, N_H),
            nn.ReLU(),
            nn.Linear(N_H, N_H),
            # MinMaxNorm(),
        )

        normalized_columns_initializer_(self.reward_output[-1].weight.data, 0.01)
        self.reward_output[-1].bias.data.fill_(0)
        normalized_columns_initializer_(self.value_logits_output[-1].weight.data, 0.01)
        self.value_logits_output[-1].bias.data.fill_(0)

    def forward(self, hidden, actions):
        x = hidden + self.action_encoder(actions)
        x, gate = self.next_state_model(x).chunk(2, -1)
        gate = gate.sigmoid()
        hidden = gate * hidden + (1 - gate) * x
        hidden = F.layer_norm(hidden, [hidden.shape[-1]])
        return self._get_output(hidden)

    @torch.jit.export
    def encode_observation(self, observation):
        hidden = self.observation_model(observation)
        hidden = F.layer_norm(hidden, [hidden.shape[-1]])
        return self._get_output(hidden)

    def _get_output(self, hidden):
        reward = self.reward_output(hidden)
        value, logits = self.value_logits_output(hidden).split_with_sizes([self.value_bins, self.logits_len], -1)
        return {'hidden': hidden, 'reward_bins': reward, 'state_value_bins': value, 'logits': logits}


class MuZero(RLBase):
    def __init__(self, observation_space, action_space,
                 reward_discount=0.99,
                 horizon=16,
                 batch_size=32,
                 num_actions=4,
                 num_rollouts=16,
                 rollout_depth=5,
                 replay_buffer_size=128*1024,
                 target_model_blend=0.005,
                 optimizer_factory=partial(optim.Adam, lr=3e-4),
                 cuda_eval=True,
                 cuda_train=True,
                 grad_clip_norm=None,
                 # reward_scale=1.0,
                 barron_alpha_c=(1.5, 1),
                 lr_scheduler_factory=None,
                 entropy_decay_factory=None,
                 **kwargs):
        super().__init__(observation_space, action_space, **kwargs)
        self._init_args = locals()
        self.reward_discount = reward_discount
        self.horizon = horizon
        self.batch_size = batch_size
        self.replay_buffer_size = replay_buffer_size
        self.target_model_blend = target_model_blend
        self.device_eval = torch.device('cuda' if cuda_eval else 'cpu')
        self.device_train = torch.device('cuda' if cuda_train else 'cpu')
        self.grad_clip_norm = grad_clip_norm
        # self.reward_scale = reward_scale
        self.barron_alpha_c = barron_alpha_c
        self.num_rollouts = num_rollouts
        self.rollout_depth = rollout_depth
        self.num_actions = num_actions

        self.pd = make_pd(action_space)
        self._train_model: MuZeroPredictionNet = torch.jit.script(MuZeroPredictionNet(
            observation_space.shape[0], self.pd.input_vector_len, self.pd.prob_vector_len))

        if self.model_init_path is not None:
            self._train_model.load_state_dict(torch.load(self.model_init_path))
            print(f'loaded model {self.model_init_path}')

        self._train_model = self._train_model.to(self.device_train)

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
            self._model_saver.check_save_model(self._train_model, self.frame_train)

    def _create_data(self):
        # (steps, actors, *)
        data = self._replay_buffer.get_last_samples(self.horizon)
        data = AttrDict(data)
        # data.rewards = self.reward_scale * data.rewards
        return data

    def _muzero_update(self, data: AttrDict):
        state_values_p1 = torch.cat([data.state_values, self._prev_data['state_values'].cpu().unsqueeze(0)], 0)
        data.value_targets = calc_value_targets(data.rewards, state_values_p1, data.dones, self.reward_discount, 0.99)

        for name, t in data.items():
            H, B = t.shape[:2]
            # (B, X, H, 1)
            image_like = t.reshape(H, B, -1).permute(1, 2, 0).unsqueeze(-1)
            X = image_like.shape[1]
            # (B, X * rollout_depth, N)
            image_like = F.unfold(image_like.float(), (self.rollout_depth, 1)).type_as(image_like)
            N = image_like.shape[-1]
            # (B, X, depth, N)
            image_like = image_like.reshape(B, X, self.rollout_depth, image_like.shape[-1])
            # (depth, B * N, *)
            data[name] = image_like.permute(2, 0, 3, 1).reshape(self.rollout_depth, B * N, *t.shape[2:])

        num_samples = data.states.shape[1]

        # rand_idx = torch.randperm(num_samples, device=self.device_train).chunk(self.num_batches)

        old_model = {k: v.clone() for k, v in self._train_model.state_dict().items()}

        # with DataLoader(data, rand_idx, self.device_train, 4, dim=1) as data_loader:
        #     for batch_index in range(self.num_batches):
        # prepare batch data
        # (H, B, *)
        # batch = AttrDict(data_loader.get_next_batch())
        # loss = self._update_traj(batch, self._do_log and batch_index == self.num_batches - 1)
        # if self._do_actor_update:
        #     blend_models(self._train_model, self._target_model, self.target_model_blend)
        # self._update_iter += 1

        loss = self._update_traj(AttrDict({k: v.to(self.device_train, non_blocking=True) for k, v in data.items()}), self._do_log)

        if self._do_log:
            self.logger.add_scalar('learning rate', self._optimizer.param_groups[0]['lr'], self.frame_train)
            self.logger.add_scalar('total loss', loss, self.frame_train)
            self.logger.add_scalar('model abs diff', model_diff(old_model, self._train_model), self.frame_train)
            self.logger.add_scalar('model max diff', model_diff(old_model, self._train_model, True), self.frame_train)

        # self._eval_model = deepcopy(self._train_model).to(self.device_eval).eval()

    # @property
    # def _do_actor_update(self):
    #     return self._update_iter % self.critic_iters == 0

    def _update_traj(self, batch, do_log=False):
        assert batch.value_targets.ndim == 2, batch.value_targets.shape
        assert batch.value_targets.shape == batch.logits.shape[:-1] == \
               batch.dones.shape == batch.rewards.shape == batch.actions.shape[:-1]

        with torch.enable_grad():
            ac_out = AttrDict(self._train_model.encode_observation(batch.states[0]))

            nonterminals = 1 - batch.dones
            vtarg = batch.value_targets[0]
            logits_target = batch.logits[0]
            ac_out_prev = ac_out
            cum_nonterm = torch.ones_like(batch.dones[0])
            loss = 0
            for i in range(batch.states.shape[0]):
                vtarg = (batch.value_targets[i] * cum_nonterm).detach()
                # vtarg = torch.lerp(vtarg, batch.value_targets[i], cum_nonterm).detach()
                logits_target = torch.lerp(logits_target, batch.logits[i], cum_nonterm.unsqueeze(-1)).detach()
                loss += -self._train_model.value_encoder.logp(ac_out.state_value_bins, vtarg).mean()
                # loss += 0.5 * (logits_target - ac_out.logits).pow(2).mean(-1).mul(cum_nonterm).mean()
                loss += 5 * self.pd.kl(logits_target, ac_out.logits).sum(-1).mul(cum_nonterm).mean()
                loss += 0.01 * -self.pd.entropy(ac_out.logits).sum(-1).mul(cum_nonterm).mean()
                if self.frame_train > 10000:
                    loss += -self.pd.logp(batch.actions[i], ac_out.logits).sum(-1).mul(cum_nonterm).mean()

                input_actions = self.pd.to_inputs(batch.actions[i])
                ac_out_prev = ac_out
                ac_out = AttrDict(self._train_model(ac_out.hidden, input_actions))
                rtarg = (batch.rewards[i] * cum_nonterm).detach()
                loss += -self._train_model.reward_encoder.logp(ac_out.reward_bins, rtarg).mean()
                cum_nonterm = cum_nonterm * nonterminals[i]

            loss = loss / batch.states.shape[0]

        if do_log:
            rewards = self._train_model.reward_encoder(ac_out.reward_bins)
            self.logger.add_scalar('reward rmse', (rewards - rtarg).pow(2).mean().sqrt(), self.frame_train)
            state_values = self._train_model.value_encoder(ac_out_prev.state_value_bins)
            self.logger.add_scalar('state_values rmse', (state_values - vtarg).pow(2).mean().sqrt(), self.frame_train)
            self.logger.add_scalar('logits rmse', (ac_out_prev.logits - logits_target).pow(2).mean().sqrt(), self.frame_train)

        loss.backward()
        # clip_grad_norm_(self._train_model.parameters(), 4)
        self._optimizer.step()
        self._optimizer.zero_grad()

        return loss

    def _search_action(self, states):
        start_ac_out = AttrDict(self._train_model.encode_observation(states))
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
            ac_out = AttrDict(self._train_model(hidden, input_actions))
            hidden = ac_out.hidden
            actions = self.pd.sample(ac_out.logits)
            rewards = self._train_model.reward_encoder(ac_out.reward_bins)
            value_targets += self.reward_discount ** i * rewards

        last_state_values = self._train_model.value_encoder(ac_out.state_value_bins)
        value_targets += self.reward_discount ** self.rollout_depth * last_state_values
        value_targets = value_targets.topk(self.num_rollouts // 4, -1)[0]
        # (B, A)
        value_targets = value_targets.mean(-1)
        assert value_targets.shape == hidden.shape[:2]
        ac_idx = value_targets.argmax(-1, keepdim=True)
        actions = start_actions.gather(-2, ac_idx.unsqueeze(-1)).squeeze(-2)
        assert actions.shape[:-1] == hidden.shape[:1]
        start_state_values = self._train_model.value_encoder(start_ac_out.state_value_bins)
        return start_ac_out.logits, actions, start_state_values

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
                self.logger.add_image('state', img, self.frame_train)
            # vsize = data.value_targets.shape[-2] ** 0.5
            # targets = data.value_targets.sum(-2) / vsize
            # values = data.state_values.sum(-2) / vsize
            # v_mean = values.mean(-1)
            # t_mean = targets.mean(-1)
            self.logger.add_scalar('entropy', self.pd.entropy(data.logits).mean(), self.frame_train)
            self.logger.add_histogram('rewards', data.rewards, self.frame_train)
            # self.logger.add_histogram('value_targets', targets, self.frame)
            # self.logger.add_histogram('advantages', data.advantages, self.frame)
            self.logger.add_histogram('values', data.state_values, self.frame_train)
            # self.logger.add_scalar('value rmse', (v_mean - t_mean).pow(2).mean().sqrt(), self.frame)
            # self.logger.add_scalar('value abs err', (v_mean - t_mean).abs().mean(), self.frame)
            # self.logger.add_scalar('value max err', (v_mean - t_mean).abs().max(), self.frame)
            if isinstance(self.pd, DiagGaussianPd):
                mean, std = data.logits.chunk(2, dim=1)
                self.logger.add_histogram('logits mean', mean, self.frame_train)
                self.logger.add_histogram('logits std', std, self.frame_train)
            elif isinstance(self.pd, CategoricalPd):
                self.logger.add_histogram('logits log_softmax', F.log_softmax(data.logits, dim=-1), self.frame_train)
            self.logger.add_histogram('logits', data.logits, self.frame_train)
            for name, param in self._train_model.named_parameters():
                self.logger.add_histogram(name, param, self.frame_train)