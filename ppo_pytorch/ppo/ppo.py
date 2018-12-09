import math
import pprint
import random
from asyncio import Future
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Optional
import os, errno

import gym.spaces
import numpy as np
import torch
import torch.autograd
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torchvision.utils import make_grid

from ..common.barron_loss import barron_loss, barron_loss_derivative
from ..common.opt_clip import opt_clip
from ..common.probability_distributions import DiagGaussianPd
from ..common.rl_base import RLBase
from ..common.pop_art import PopArt
from ..common.attr_dict import AttrDict
from ..common.data_loader import DataLoader
from ..models import create_ppo_fc_actor, Actor
from ..models.heads import PolicyHead, StateValueHead, StateValueQuantileHead
from ..models.utils import model_diff
from .steps_processor import StepsProcessor
from ..common.target_logits import get_target_logits
from optfn.iqn_loss import huber_quantile_loss


class PPO(RLBase):
    def __init__(self, observation_space, action_space,
                 reward_discount=0.99,
                 advantage_discount=0.95,
                 horizon=64,
                 ppo_iters=10,
                 batch_size=64,
                 model_factory=create_ppo_fc_actor,
                 optimizer_factory=partial(optim.Adam, lr=3e-4),
                 value_loss_scale=0.5,
                 entropy_loss_scale=0.01,
                 entropy_reward_scale=0.0,
                 constraint='clip',
                 policy_clip=0.1,
                 value_clip=0.1,
                 kl_target=0.01,
                 kl_scale=0.1,
                 lr_iter_mult=1.0,
                 cuda_eval=False,
                 cuda_train=False,
                 grad_clip_norm=2,
                 reward_scale=1.0,
                 barron_alpha_c=(1.5, 1),
                 advantage_scaled_clip=True,
                 lr_scheduler_factory=None,
                 clip_decay_factory=None,
                 entropy_decay_factory=None,
                 model_save_folder='./models',
                 model_save_tag='ppo_model',
                 model_save_interval=100_000,
                 model_init_path=None,
                 save_intermediate_models=False,
                 use_pop_art=False,
                 **kwargs):
        """
        Single threaded implementation of Proximal Policy Optimization Algorithms
        https://arxiv.org/pdf/1707.06347.pdf

        Tis implementation have several differences from PPO paper.
        1)  State-value is optimized with Barron loss. Advantages are scaled using Barron loss derivative.
            To use MSE loss for state-value and unscaled advantages set `barron_alpha_c` to (2, 1).
            A More General Robust Loss Function https://arxiv.org/abs/1701.03077
        2)  Policy / value clip constraint is multiplied by abs(advantages).
            This will make constraint different for each element in batch.
            Set `advantage_scaled_clip` to false to disable.
        3)  KL Divergence penalty implementation is different.
            When `kl` < `kl_target` it is not applied.
            When `kl` > `kl_target` it is scaled quadratically based on abs(`kl` - `kl_target`)
                and policy and entropy maximization objectives are disabled.
        4)  New constraint type which clips raw network output vector instead of action log logits.
            See 'opt' in `constraint` documentation.
        4)  Several different constraints could be applied at same time.

        Args:
            observation_space (gym.Space): Environment's observation space
            action_space (gym.Space): Environment's action space
            reward_discount (float): Value function discount factor
            advantage_discount (float): Global Advantage Estimation discount factor
            horizon (int): Training will happen each `horizon` * `num_actors` steps
            ppo_iters (int): Number training passes for each state
            batch_size (int): Training batch size
            model_factory (Callable[[gym.Space, gym.Space], nn.Model]):
                Callable object receiving (observation_space, action_space) and returning actor-critic model
            optimizer_factory (Callable[[List[nn.Parameter]], optim.Optimizer]):
                Callable object receiving `model.parameters()` and returning model optimizer
            value_loss_scale (float): Multiplier for state-value loss
            entropy_loss_scale (float): Entropy maximization loss bonus (typically 0 to 0.01)
            entropy_reward_scale (float): Scale for additional reward based on entropy (typically 0 to 0.5)
            constraint tuple[str]: Policy optimization constraint. State value always uses 'clip' constraint.
                Tuple could contain zero or more of these values:
                'clip' - PPO clipping. Implementation is somewhat different from PPO paper.
                    Controlled by `policy_clip` and `value_clip`,
                'kl' - KL Divergence based constraint, implementation is very different from PPO paper.
                    Controlled by `kl_target` and `kl_scale`
                'opt' - clip raw action logits and state-value.
                    Controlled by `policy_clip` and `value_clip`,
            policy_clip (float): policy clip strength
            value_clip (float): State-value clip strength
            kl_target (float): Desired KL Divergence for 'kl' policy penalty (typically 0.001 to 0.03)
            kl_scale (float): KL penalty multiplier
            cuda_eval (bool): Use CUDA for environment steps
            cuda_train (bool): Use CUDA for training steps
            grad_clip_norm (float or None): Max norm for gradient clipping (typically 0.5 to 40)
            reward_scale (float): Scale factor for environment's rewards
            barron_alpha_c (float, float): Coefficients 'alpha' and 'c' for loss function proposed in
                A More General Robust Loss Function https://arxiv.org/abs/1701.03077
                Default (1, 1.5) will give something in between MSE and pseudo Huber.
            advantage_scaled_clip (bool): Whether to multiply `policy_clip` and `value_clip` by abs(advantages)
            lr_scheduler_factory (Callable[DecayLR]): Learning rate scheduler factory.
            clip_decay_factory (Callable[ValueDecay]): Policy / value clip scheduler factory.
            entropy_decay_factory (Callable[ValueDecay]): `entropy_loss_scale` scheduler factory.
            model_save_folder (str): Directory where models will be saved.
            model_save_tag (str): Tag is added to name of saved model. Used to save different models in one folder.
            model_save_interval (int): Interval in frames between model saves.
                Set to None to disable model saving.
            model_init_path (str): Path to model file to init from.
            save_intermediate_models (bool): If True, model saved at each `model_save_interval` frame
                is saved alongside new model. Otherwise it is overwritten by new model.
            num_actors (int): Number of parallel environments
            log_time_interval (float): Tensorboard logging interval in seconds
        """
        super().__init__(observation_space, action_space, **kwargs)
        self._init_args = locals()
        self.reward_discount = reward_discount
        self.advantage_discount = advantage_discount
        self.policy_clip = policy_clip
        self.value_clip = value_clip
        self.entropy_loss_scale = entropy_loss_scale
        self.horizon = horizon
        self.ppo_iters = ppo_iters
        self.batch_size = batch_size
        self.device_eval = torch.device('cuda' if cuda_eval else 'cpu')
        self.device_train = torch.device('cuda' if cuda_train else 'cpu')
        self.grad_clip_norm = grad_clip_norm
        self.value_loss_scale = value_loss_scale
        self.model_factory = model_factory
        self.constraint = (constraint,) if isinstance(constraint, str) else constraint
        self.reward_scale = reward_scale
        self.model_save_folder = model_save_folder
        self.model_save_interval = model_save_interval
        self.save_intermediate_models = save_intermediate_models
        self.model_save_tag = model_save_tag
        self.kl_target = kl_target
        self.kl_scale = kl_scale
        self.lr_iter_mult = lr_iter_mult
        self.entropy_reward_scale = entropy_reward_scale
        self.barron_alpha_c = barron_alpha_c
        self.advantage_scaled_clip = advantage_scaled_clip
        self.use_pop_art = use_pop_art
        self.num_quantiles = 16

        assert len(set(self.constraint) - {'clip', 'kl', 'opt', 'mse', 'target'}) == 0

        if model_init_path is None:
            self._train_model: Actor = model_factory(observation_space, action_space)
        else:
            self._train_model: Actor = torch.load(model_init_path)
            print(f'loaded model {model_init_path}')
        self._optimizer = optimizer_factory(self._train_model.parameters())
        self._lr_scheduler = lr_scheduler_factory(self._optimizer) if lr_scheduler_factory is not None else None
        self._clip_decay = clip_decay_factory() if clip_decay_factory is not None else None
        self._entropy_decay = entropy_decay_factory() if entropy_decay_factory is not None else None
        self._last_model_save_frame = 0
        self._pop_art = PopArt()
        self._steps_processor = self._create_steps_processor()
        self._train_future: Optional[Future] = None
        self._train_executor = ThreadPoolExecutor(max_workers=1)
        self._train_model = self._train_model.to(self.device_eval, non_blocking=True).train()
        self._eval_model = deepcopy(self._train_model).to(self.device_eval).eval()
        self._prev_value_tau = None

    def _step(self, rewards, dones, states) -> torch.Tensor:
        with torch.no_grad():
            # run network
            states_eval = states.to(self.device_eval)

            value_head = self._eval_model.heads.state_values
            iqn = isinstance(value_head, StateValueQuantileHead)
            if iqn:
                cur_tau = torch.rand((states.shape[0], self.num_quantiles), device=self.device_eval)
                prev_value, prev_tau = (torch.zeros_like(cur_tau), torch.full_like(cur_tau, 0.5)) \
                    if self._prev_value_tau is None else self._prev_value_tau
                if dones is not None:
                    prev_value[dones > 0.5] = 0
                    prev_tau[dones > 0.5] = 0.5
                tau_values = torch.cat([cur_tau, prev_tau, prev_value], -1)
                tau = dict(tau=tau_values)
            else:
                tau = dict()

            ac_out = self._take_step(states_eval, dones, **tau)
            actions = self._eval_model.heads.logits.pd.sample(ac_out.logits).cpu()

            if not self.disable_training:
                if iqn:
                    self._prev_value_tau = (ac_out.state_values, cur_tau)

                self._steps_processor.append_values(states=states, rewards=rewards, dones=dones, actions=actions, **ac_out, **tau)

                if len(self._steps_processor.data.states) > self.horizon:
                    self._pre_train()
                    self._train()

            return actions

    def _take_step(self, states, dones, **model_params):
        return self._eval_model(states, **model_params)

    def _pre_train(self):
        self._check_log()

        # update clipping and learning rate decay schedulers
        if self._lr_scheduler is not None:
            self._lr_scheduler.step(self.frame)
        if self._clip_decay is not None:
            self._clip_decay.step(self.frame)
        if self._entropy_decay is not None:
            self._entropy_decay.step(self.frame)

    def _train(self):
        data = self._create_data()
        self._train_async(data)
        # if self._train_future is not None:
        #     self._train_future.result()
        # self._train_future = self._train_executor.submit(self._train_async, data)

    def _create_data(self):
        self._steps_processor.complete()
        data = self._steps_processor.data
        self._steps_processor = self._create_steps_processor()
        return data

    def _train_async(self, data):
        with torch.no_grad():
            self._log_training_data(data)
            self._ppo_update(data)
            self._check_save_model()

    def _ppo_update(self, data: AttrDict):
        value_head = self._train_model.heads.state_values
        iqn = isinstance(value_head, StateValueQuantileHead)

        if self.use_pop_art:
            returns_mean, returns_std = self._pop_art.statistics
            self._pop_art.update_statistics(data.returns)
            value_head.normalize(returns_mean, returns_std)
            data.state_values = (data.state_values - returns_mean) / returns_std
            data.returns = (data.returns - returns_mean) / returns_std

        tau = dict(tau=data.tau) if iqn else dict()

        data = AttrDict(states=data.states, logits_old=data.logits, state_values_old=data.state_values,
                        actions=data.actions, advantages=data.advantages, returns=data.returns, **tau)

        if 'target' in self.constraint:
            data.logits_target = get_target_logits(self._train_model.heads.logits.pd, data.actions, data.logits_old, self.policy_clip * data.advantages)

        batches = max(1, math.ceil(self.num_actors * self.horizon / self.batch_size))

        initial_lr = [g['lr'] for g in self._optimizer.param_groups]

        rand_idx = torch.randperm(len(data.state_values_old) * self.ppo_iters, device=self.device_train)
        rand_idx = rand_idx.fmod_(len(data.state_values_old)).chunk(batches * self.ppo_iters)

        old_model = deepcopy(self._train_model)
        kl_list = []

        with DataLoader(data, rand_idx, self.device_train, 4) as data_loader:
            for ppo_iter in range(self.ppo_iters):
                for loader_iter in range(batches):
                    # prepare batch data
                    batch = AttrDict(data_loader.get_next_batch())
                    loss, kl = self._ppo_step(batch, self._do_log and ppo_iter == self.ppo_iters - 1 and loader_iter == 0)
                    kl_list.append(kl)

                for g in self._optimizer.param_groups:
                    g['lr'] *= self.lr_iter_mult

        for g, lr in zip(self._optimizer.param_groups, initial_lr):
            g['lr'] = lr

        kl = np.mean(kl_list)

        if self._do_log:
            self.logger.add_scalar('learning rate', self._learning_rate, self.frame)
            self.logger.add_scalar('clip mult', self._clip_mult, self.frame)
            self.logger.add_scalar('total loss', loss, self.frame)
            self.logger.add_scalar('kl', kl, self.frame)
            self.logger.add_scalar('kl scale', self.kl_scale, self.frame)
            self.logger.add_scalar('model abs diff', model_diff(old_model, self._train_model), self.frame)
            self.logger.add_scalar('model max diff', model_diff(old_model, self._train_model, True), self.frame)

        if self.use_pop_art:
            returns_mean, returns_std = self._pop_art.statistics
            self._train_model.heads.state_values.unnormalize(returns_mean, returns_std)

        self._adjust_kl_scale(kl)

        self._copy_parameters(self._train_model, self._eval_model)
        # self._eval_model = deepcopy(self._train_model).to(self.device_eval).eval()

    def _ppo_step(self, batch, do_log):
        with torch.enable_grad():
            value_head = self._train_model.heads.state_values
            iqn = isinstance(value_head, StateValueQuantileHead)

            actor_params = AttrDict()
            if do_log:
                actor_params.logger = self.logger
                actor_params.cur_step = self.step
            if iqn:
                actor_params.tau = batch.tau

            actor_out = self._train_model(batch.states, **actor_params)

            batch.logits = actor_out.logits
            batch.state_values = actor_out.state_values

            loss, kl = self._get_ppo_loss(batch, do_log=do_log)
            loss = loss.mean()
        kl = kl.item()
        if 'kl' not in self.constraint and 'mse' not in self.constraint or kl < 4 * self.kl_target:
            # optimize
            loss.backward()
            if self.grad_clip_norm is not None:
                clip_grad_norm_(self._train_model.parameters(), self.grad_clip_norm)
            self._optimizer.step()
            self._optimizer.zero_grad()

        return loss, kl

    def _get_ppo_loss(self, batch, pd=None, do_log=False, tag=''):
        """
        Single iteration of PPO algorithm.
        Returns: Total loss and KL divergence.
        """

        logits, logits_old = batch.logits, batch.logits_old
        values, values_old = batch.state_values, batch.state_values_old
        returns = batch.returns
        actions = batch.actions
        advantages = batch.advantages

        if pd is None:
            pd = self._train_model.heads.logits.pd

        # clipping factors
        value_clip = self.value_clip * self._clip_mult
        policy_clip = self.policy_clip * self._clip_mult

        if 'opt' in self.constraint:
            logits = opt_clip(logits, logits_old, policy_clip)
            values = opt_clip(values, values_old, value_clip)

        # action probability ratio
        # log probabilities used for better numerical stability
        logp_old = pd.logp(actions, logits_old)
        logp = pd.logp(actions, logits)
        ratio = logp - logp_old

        adv_u = advantages.unsqueeze(-1)

        if 'clip' in self.constraint:
            unclipped_policy_loss = ratio * adv_u
            if self.advantage_scaled_clip:
                pclip = adv_u.abs() * policy_clip
                clipped_ratio = torch.min(torch.max(ratio, -pclip), pclip)
            else:
                clipped_ratio = ratio.clamp(-policy_clip, policy_clip)
            clipped_policy_loss = clipped_ratio * adv_u
            loss_clip = -torch.min(unclipped_policy_loss, clipped_policy_loss)
        elif 'target' in self.constraint:
            diff_cur = batch.logits_target - logits
            diff_old = batch.logits_target - logits_old
            diff_old_rms = diff_old.pow(2).mean().sqrt().clamp(min=1e-3)

            # loss_clip = diff_cur * diff_old / diff_old_rms
            # loss_clip = adv_u.abs() * loss_clip.clamp(min=0)

            loss_clip = diff_cur * diff_cur.detach() / diff_old_rms
            loss_clip = adv_u.abs() * loss_clip
        else:
            # unclipped loss
            loss_clip = -ratio * adv_u

        # entropy bonus for better exploration
        entropy = pd.entropy(logits)
        loss_ent = -self.entropy_loss_scale * entropy

        kl = pd.kl(logits_old, logits)
        if 'kl' in self.constraint:
            loss_kl = self.kl_scale * (kl + 10 * (kl - 2 * self.kl_target).clamp(0, 1e6).pow(2))
        elif 'mse' in self.constraint:
            loss_kl = self.kl_scale * (logits - logits_old).abs().pow(2.5)
        else:
            loss_kl = kl.new(1).zero_()

        # value loss
        if 'tau' in batch:
            tau = batch.tau[..., :self.num_quantiles]
            loss_value = self.value_loss_scale * huber_quantile_loss(values, returns, tau, reduce=False)
            # num_q_per_batch = max(1, int(round(self.num_quantiles / self.ppo_iters)))
            # idx = torch.randperm(self.num_quantiles, device=values.device)[:num_q_per_batch]
            # loss_value = self.value_loss_scale * huber_quantile_loss(values[..., idx], returns[..., idx], tau[..., idx], reduce=False)
        else:
            v_pred_clipped = values_old + (values - values_old).clamp(-value_clip, value_clip)
            vf_clip_loss = barron_loss(v_pred_clipped, returns, *self.barron_alpha_c, reduce=False)
            vf_nonclip_loss = barron_loss(values, returns, *self.barron_alpha_c, reduce=False)
            loss_value = self.value_loss_scale * torch.max(vf_nonclip_loss, vf_clip_loss)

        loss_value = loss_value.mean(-1)
        loss_clip = loss_clip.mean(-1)
        loss_ent = loss_ent.mean(-1)
        loss_kl = loss_kl.mean(-1)

        assert loss_clip.shape == loss_value.shape, (loss_clip.shape, loss_value.shape)
        assert loss_value.shape == loss_ent.shape, (loss_value.shape, loss_ent.shape)
        assert loss_ent.shape == loss_kl.shape or 'kl' not in self.constraint, (loss_ent.shape, loss_kl.shape)
        # sum all losses
        total_loss = loss_clip + loss_value + loss_kl + loss_ent
        assert not np.isnan(total_loss.mean().item()) and not np.isinf(total_loss.mean().item()), \
            (loss_clip.mean().item(), loss_value.mean().item(), loss_ent.mean().item())

        if do_log and tag is not None:
            with torch.no_grad():
                self.logger.add_scalar('entropy' + tag, entropy.mean(), self.frame)
                self.logger.add_scalar('loss entropy' + tag, loss_ent.mean(), self.frame)
                self.logger.add_scalar('loss state value' + tag, loss_value.mean(), self.frame)
                self.logger.add_scalar('ratio mean' + tag, ratio.mean(), self.frame)
                self.logger.add_scalar('ratio abs mean' + tag, ratio.abs().mean(), self.frame)
                self.logger.add_scalar('ratio abs max' + tag, ratio.abs().max(), self.frame)
                self.logger.add_scalar('loss policy' + tag, loss_clip.mean(), self.frame)

        return total_loss, kl.mean()

    def _adjust_kl_scale(self, kl):
        threshold, change, limit = 1.2, 1.1, 100.0
        if kl > threshold * self.kl_target:
            self.kl_scale = min(limit, self.kl_scale * change)
        if kl < (1 / threshold) * self.kl_target:
            self.kl_scale = max(1 / limit, self.kl_scale / change)

    # @staticmethod
    # def _head_factory(hidden_size, pd):
    #     return dict(logits=PolicyHead(hidden_size, pd), state_values=StateValueQuantileHead(hidden_size))

    @property
    def _learning_rate(self):
        return self._optimizer.param_groups[0]['lr']

    @property
    def _clip_mult(self):
        return self._clip_decay.value if self._clip_decay is not None else 1

    def _log_set(self):
        self.logger.add_text(self.__class__.__name__, pprint.pformat(self._init_args))
        self.logger.add_text('Model', str(self._train_model))

    def drop_collected_steps(self):
        self._steps_processor = self._create_steps_processor()

    def _check_save_model(self):
        if self.model_save_interval is None or \
           self._last_model_save_frame + self.model_save_interval > self.frame:
            return

        self._create_save_folder()
        path = self._get_save_path()
        self._save_model(path)

    def _save_model(self, path):
        # print(f'saving model at {self.frame} step to {path}')
        model = deepcopy(self._train_model).cpu()
        try:
            torch.save(model, path)
        except OSError as e:
            print('error while saving model', e)

    def _get_save_path(self):
        self._last_model_save_frame = self.frame
        if self.save_intermediate_models:
            name = f'{self.model_save_tag}_{self.actor_index}_{self.frame}'
        else:
            name = f'{self.model_save_tag}_{self.actor_index}'
        return Path(self.model_save_folder) / (name + '.pth')

    def _create_save_folder(self):
        try:
            os.makedirs(self.model_save_folder)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

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
            self.logger.add_histogram('rewards', data.rewards, self.frame)
            self.logger.add_histogram('returns', data.returns, self.frame)
            self.logger.add_histogram('advantages', data.advantages, self.frame)
            self.logger.add_histogram('values', data.state_values, self.frame)
            self.logger.add_scalar('value rmse', (data.state_values - data.returns).pow(2).mean().sqrt(), self.frame)
            self.logger.add_scalar('value abs err', (data.state_values - data.returns).abs().mean(), self.frame)
            self.logger.add_scalar('value max err', (data.state_values - data.returns).max(), self.frame)
            if isinstance(self._train_model.heads.logits.pd, DiagGaussianPd):
                mean, std = data.logits.chunk(2, dim=1)
                self.logger.add_histogram('logits mean', mean, self.frame)
                self.logger.add_histogram('logits std', std, self.frame)
            else:
                self.logger.add_histogram('logits', F.log_softmax(data.logits, dim=-1), self.frame)
            for name, param in self._train_model.named_parameters():
                self.logger.add_histogram(name, param, self.frame)

    def _create_steps_processor(self) -> StepsProcessor:
        return StepsProcessor(self._train_model.heads.logits.pd, self.reward_discount, self.advantage_discount,
                              self.reward_scale, True, self.barron_alpha_c, self.entropy_reward_scale)

    def _copy_parameters(self, src, dst):
        for src, dst in zip(src.state_dict().values(), dst.state_dict().values()):
            dst.data.copy_(src.data)

    def __getstate__(self):
        d = dict(self.__dict__)
        d['_logger'] = None
        return d

    def __setstate__(self, d):
        self.__dict__ = d
