import math
import pprint
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

from ..common.barron_loss import barron_loss
from ..common.opt_clip import opt_clip
from ..common.probability_distributions import DiagGaussianPd
from ..common.rl_base import RLBase
from ..common.pop_art import PopArt
from ..common.attr_dict import AttrDict
from ..common.data_loader import DataLoader
from ..models import FCActor, Actor
from ..models.heads import PolicyHead, StateValueHead
from ..models.utils import model_diff
from .steps_processor import StepsProcessor


class PPO(RLBase):
    def __init__(self, observation_space, action_space,
                 reward_discount=0.99,
                 advantage_discount=0.95,
                 horizon=64,
                 ppo_iters=10,
                 batch_size=64,
                 model_factory=FCActor,
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
                 hidden_code_type: 'input' or 'first' or 'last'='input',
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
        4)  New constraint type which clips raw network output vector instead of action log probs.
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
                'opt' - clip raw action probs and state-value.
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
            hidden_code_type (str): Model hidden code type. Not used in PPO.
                Valid values are 'input' or 'first' or 'last'
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
        self.hidden_code_type = hidden_code_type
        self.barron_alpha_c = barron_alpha_c
        self.advantage_scaled_clip = advantage_scaled_clip
        self.use_pop_art = use_pop_art

        assert len(set(self.constraint) - {'clip', 'kl', 'opt', 'mse'}) == 0

        if model_init_path is None:
            self._train_model: Actor = model_factory(observation_space, action_space, self._head_factory, hidden_code_type=hidden_code_type)
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

    def _step(self, prev_states, rewards, dones, cur_states) -> torch.Tensor:
        with torch.no_grad():
            # run network
            ac_out = self._take_step(cur_states.to(self.device_eval), dones)
            actions = self._eval_model.pd.sample(ac_out.probs).cpu()

            if not self.disable_training:
                self._steps_processor.append_head(ac_out, states=cur_states, rewards=rewards, dones=dones, actions=actions)

                if len(self._steps_processor.data.states) > self.horizon:
                    self._pre_train()
                    self._train()

            return actions

    def _take_step(self, states, dones):
        return self._eval_model(states)

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
        self._steps_processor.complete()
        data = self._steps_processor.data
        self._steps_processor = self._create_steps_processor()

        self._train_async(data)
        # if self._train_future is not None:
        #     self._train_future.result()
        # self._train_future = self._train_executor.submit(self._train_async, data)

    def _train_async(self, data):
        with torch.no_grad():
            self._log_training_data(data)
            self._ppo_update(data)
            self._check_save_model()

    def _ppo_update(self, data: AttrDict):
        if self.use_pop_art:
            returns_mean, returns_std = self._pop_art.statistics
            self._pop_art.update_statistics(data.returns)
            self._train_model.heads.state_values.normalize(returns_mean, returns_std)
        else:
            returns_mean, returns_std = 0, 1

        data = [data.states, data.probs, (data.state_values - returns_mean) / returns_std,
                data.actions, data.advantages, (data.returns - returns_mean) / returns_std]

        batches = max(1, math.ceil(self.num_actors * self.horizon / self.batch_size))

        initial_lr = [g['lr'] for g in self._optimizer.param_groups]

        rand_idx = torch.randperm(len(data[0]) * self.ppo_iters, device=self.device_train)
        rand_idx = rand_idx.fmod_(len(data[0])).chunk(batches * self.ppo_iters)

        old_model = deepcopy(self._train_model)

        with DataLoader(data, rand_idx, self.device_train, 4) as data_loader:
            for ppo_iter in range(self.ppo_iters):
                for loader_iter in range(batches):
                    # prepare batch data
                    st, po, vo, ac, adv, ret = data_loader.get_next_batch()
                    if ppo_iter == self.ppo_iters - 1 and loader_iter == 0:
                        self._train_model.set_log(self.logger, self._do_log, self.step)

                    loss, kl = self._ppo_step(st, po, vo, ac, adv, ret)

                    self._train_model.set_log(self.logger, False, self.step)

                for g in self._optimizer.param_groups:
                    g['lr'] *= self.lr_iter_mult

        for g, lr in zip(self._optimizer.param_groups, initial_lr):
            g['lr'] = lr

        if self._do_log:
            self.logger.add_scalar('learning rate', self._learning_rate, self.frame)
            self.logger.add_scalar('clip mult', self._clip_mult, self.frame)
            self.logger.add_scalar('total loss', loss, self.frame)
            self.logger.add_scalar('kl', kl, self.frame)
            self.logger.add_scalar('model abs diff', model_diff(old_model, self._train_model), self.frame)
            self.logger.add_scalar('model max diff', model_diff(old_model, self._train_model, True), self.frame)

        if self.use_pop_art:
            returns_mean, returns_std = self._pop_art.statistics
            self._train_model.heads.state_values.unnormalize(returns_mean, returns_std)

        self._eval_model = deepcopy(self._train_model).to(self.device_eval).eval()

    def _ppo_step(self, states, probs_old, values_old, actions, advantages, returns):
        with torch.enable_grad():
            actor_out = self._train_model(states)
            probs = actor_out.probs
            values = actor_out.state_values
            # get loss
            loss, kl = self._get_ppo_loss(probs, probs_old, values, values_old, actions, advantages, returns)
            loss = loss.mean()

        # optimize
        loss.backward()
        if self.grad_clip_norm is not None:
            clip_grad_norm_(self._train_model.parameters(), self.grad_clip_norm)
        self._optimizer.step()
        self._optimizer.zero_grad()

        return loss, kl

    def _get_ppo_loss(self, probs, probs_old, values, values_old, actions, advantages, returns, pd=None, tag=''):
        """
        Single iteration of PPO algorithm.
        Returns: Total loss and KL divergence.
        """

        if pd is None:
            pd = self._train_model.pd

        # clipping factors
        value_clip = self.value_clip * self._clip_mult
        policy_clip = self.policy_clip * self._clip_mult

        if 'opt' in self.constraint:
            probs = opt_clip(probs, probs_old, policy_clip)
            values = opt_clip(values, values_old, value_clip)

        # action probability ratio
        # log probabilities used for better numerical stability
        logp_old = pd.logp(actions, probs_old)
        logp = pd.logp(actions, probs)
        ratio = logp - logp_old.detach()

        adv_u = advantages.unsqueeze(-1)

        unclipped_policy_loss = ratio * adv_u
        if 'clip' in self.constraint:
            if self.advantage_scaled_clip:
                pclip = adv_u.abs() * policy_clip
                clipped_ratio = torch.min(torch.max(ratio, -pclip), pclip)
            else:
                clipped_ratio = ratio.clamp(-policy_clip, policy_clip)
            clipped_policy_loss = clipped_ratio * adv_u
            loss_clip = -torch.min(unclipped_policy_loss, clipped_policy_loss)
        else:
            # unclipped loss
            loss_clip = -unclipped_policy_loss

        # entropy bonus for better exploration
        entropy = pd.entropy(probs)
        loss_ent = -self.entropy_loss_scale * entropy

        kl = pd.kl(probs_old, probs)
        if 'kl' in self.constraint:
            kl_targets = self.kl_target * adv_u.abs()
            loss_kl = (kl - kl_targets).div(self.kl_target).pow(2).mul(self.kl_scale * self.kl_target)
            small_kl = (kl < self.kl_target).detach()
            large_kl = (kl > self.kl_target).detach()
            loss_kl[small_kl] = 0
            loss_ent[large_kl] = 0
            loss_clip[large_kl] = 0
        elif 'mse' in self.constraint:
            loss_kl = self.kl_scale * 10 * (probs - probs_old).abs().pow(2.5)
        else:
            loss_kl = kl.new(1).zero_()

        loss_clip = loss_clip.mean(-1)
        loss_ent = loss_ent.mean(-1)
        loss_kl = loss_kl.mean(-1)

        # value loss
        if self.advantage_scaled_clip:
            vclip = advantages.abs() * value_clip
            v_pred_clipped = values_old + torch.min(torch.max(values - values_old, -vclip), vclip)
        else:
            v_pred_clipped = values_old + (values - values_old).clamp(-value_clip, value_clip)
        vf_clip_loss = barron_loss(v_pred_clipped, returns, *self.barron_alpha_c, reduce=False)
        vf_nonclip_loss = barron_loss(values, returns, *self.barron_alpha_c, reduce=False)
        loss_value = self.value_loss_scale * torch.max(vf_nonclip_loss, vf_clip_loss)

        assert loss_clip.shape == loss_value.shape, (loss_clip.shape, loss_value.shape)
        assert loss_value.shape == loss_ent.shape, (loss_value.shape, loss_ent.shape)
        assert loss_ent.shape == loss_kl.shape or 'kl' not in self.constraint, (loss_ent.shape, loss_kl.shape)
        # sum all losses
        total_loss = loss_clip + loss_value + loss_kl + loss_ent
        assert not np.isnan(total_loss.mean().item()) and not np.isinf(total_loss.mean().item()), \
            (loss_clip.mean().item(), loss_value.mean().item(), loss_ent.mean().item())

        if self._train_model.do_log and tag is not None:
            with torch.no_grad():
                self.logger.add_histogram('loss value' + tag, loss_value, self.frame)
                self.logger.add_histogram('loss ent' + tag, loss_ent, self.frame)
                self.logger.add_scalar('entropy' + tag, entropy.mean(), self.frame)
                self.logger.add_scalar('loss entropy' + tag, loss_ent.mean(), self.frame)
                self.logger.add_scalar('loss value' + tag, loss_value.mean(), self.frame)
                self.logger.add_histogram('ratio' + tag, ratio, self.frame)
                self.logger.add_scalar('ratio mean' + tag, ratio.mean(), self.frame)
                self.logger.add_scalar('ratio abs mean' + tag, ratio.abs().mean(), self.frame)
                self.logger.add_scalar('ratio abs max' + tag, ratio.abs().max(), self.frame)
                if 'clip' in self.constraint:
                    self.logger.add_histogram('loss clip' + tag, loss_clip, self.frame)
                    self.logger.add_scalar('loss clip' + tag, loss_clip.mean(), self.frame)

        return total_loss, kl.mean()

    @staticmethod
    def _head_factory(hidden_size, pd):
        return dict(probs=PolicyHead(hidden_size, pd), state_values=StateValueHead(hidden_size))

    @property
    def _learning_rate(self):
        return self._optimizer.param_groups[0]['lr']

    @property
    def _clip_mult(self):
        return self._clip_decay.value if self._clip_decay is not None else 1

    def _log_set(self):
        self.logger.add_text('PPO', pprint.pformat(self._init_args))
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
            if isinstance(self._train_model.pd, DiagGaussianPd):
                mean, std = data.probs.chunk(2, dim=1)
                self.logger.add_histogram('probs mean', mean, self.frame)
                self.logger.add_histogram('probs std', std, self.frame)
            else:
                self.logger.add_histogram('probs', F.log_softmax(data.probs, dim=-1), self.frame)
            for name, param in self._train_model.named_parameters():
                self.logger.add_histogram(name, param, self.frame)

    def _create_steps_processor(self) -> StepsProcessor:
        return StepsProcessor(self._train_model.pd, self.reward_discount, self.advantage_discount,
                              self.reward_scale, True, self.barron_alpha_c, self.entropy_reward_scale)

    def __getstate__(self):
        d = dict(self.__dict__)
        d['_logger'] = None
        return d

    def __setstate__(self, d):
        self.__dict__ = d
