import math
import pprint
from copy import deepcopy
from enum import Enum
from functools import partial
from typing import Optional

import gym.spaces
import numpy as np
import torch
import torch.autograd
import torch.nn.functional as F
import torch.optim as optim
from rl_exp.noisy_linear import NoisyLinear
from torch.nn.utils import clip_grad_norm_
from torchvision.utils import make_grid

from .steps_processor import StepsProcessor
from ..actors import create_ppo_fc_actor, Actor
from ..actors.utils import model_diff
from ..common.attr_dict import AttrDict
from ..common.barron_loss import barron_loss
from ..common.data_loader import DataLoader
from ..common.pop_art import PopArt
from ..common.probability_distributions import DiagGaussianPd, CategoricalPd, ProbabilityDistribution
from ..common.rl_base import RLBase


class Constraint(Enum):
    clip = 'clip'
    spu = 'spu'
    # v_mpo = 'v_mpo'
    kl = 'kl'
    target = 'target'
    
    
class SchedulerManager:
    def __init__(self, optimizer,
                 lr_scheduler_factory=None,
                 clip_decay_factory=None,
                 entropy_decay_factory=None):
        self._lr_scheduler = lr_scheduler_factory(optimizer) if lr_scheduler_factory is not None else None
        self._clip_decay = clip_decay_factory() if clip_decay_factory is not None else None
        self._entropy_decay = entropy_decay_factory() if entropy_decay_factory is not None else None

    @property
    def clip_decay(self):
        return self._clip_decay.value if self._clip_decay is not None else 1
    
    def step(self, current_frame):
        # update clipping and learning rate decay schedulers
        if self._lr_scheduler is not None:
            self._lr_scheduler.step(current_frame)
        if self._clip_decay is not None:
            self._clip_decay.step(current_frame)
        if self._entropy_decay is not None:
            self._entropy_decay.step(current_frame)


def copy_state_dict(src, dst):
    for src, dst in zip(src.state_dict().values(), dst.state_dict().values()):
        dst.data.copy_(src.data)


def log_training_data(do_log, logger, frame_train, train_model, data: AttrDict):
    if do_log:
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
            logger.add_image('state', img, frame_train)
        targets = data.value_targets
        values = data.state_values
        logger.add_histogram('Rewards', data.rewards, frame_train)
        logger.add_histogram('Value Targets', targets, frame_train)
        logger.add_histogram('Advantages', data.advantages, frame_train)
        logger.add_histogram('Values', values, frame_train)
        logger.add_scalar('Value Errors/RMSE', (values - targets).pow(2).mean().sqrt(), frame_train)
        logger.add_scalar('Value Errors/Abs', (values - targets).abs().mean(), frame_train)
        logger.add_scalar('Value Errors/Max', (values - targets).abs().max(), frame_train)
        if isinstance(train_model.heads.logits.pd, DiagGaussianPd):
            mean, std = data.logits.chunk(2, dim=1)
            logger.add_histogram('Logits Mean', mean, frame_train)
            logger.add_histogram('Logits Std', std, frame_train)
        elif isinstance(train_model.heads.logits.pd, CategoricalPd):
            logger.add_histogram('Logits Log Softmax', F.log_softmax(data.logits, dim=-1), frame_train)
        logger.add_histogram('Logits Logits', data.logits, frame_train)
        for name, param in train_model.named_parameters():
            logger.add_histogram(name, param, frame_train)


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
                 spu_dis_agg_lam=(0.01, 0.01 / 1.3, 1.0),
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
        self.optimizer_factory = optimizer_factory
        self.reward_scale = reward_scale
        self.kl_target = kl_target
        self.kl_scale = self._init_kl_scale = kl_scale
        self.lr_iter_mult = lr_iter_mult
        self.entropy_reward_scale = entropy_reward_scale
        self.barron_alpha_c = barron_alpha_c
        self.advantage_scaled_clip = advantage_scaled_clip
        self.spu_dis_agg_lam = spu_dis_agg_lam
        self.use_pop_art = use_pop_art
        self.lr_scheduler_factory = lr_scheduler_factory

        assert isinstance(constraint, str) or isinstance(constraint, list) or isinstance(constraint, tuple)
        self.constraint = (constraint,) if isinstance(constraint, str) else constraint
        self.constraint = [Constraint[c] for c in self.constraint]
        assert len(set(self.constraint) - set(c for c in Constraint)) == 0

        self._train_model: Actor = model_factory(observation_space, action_space)
        self._eval_model: Actor = model_factory(observation_space, action_space)
        if self.model_init_path is not None:
            self._train_model.load_state_dict(torch.load(self.model_init_path), True)
            print(f'loaded model {self.model_init_path}')
        copy_state_dict(self._train_model, self._eval_model)
        self._train_model = self._train_model.train().to(self.device_train, non_blocking=True)
        self._eval_model = self._eval_model.eval().to(self.device_eval, non_blocking=True)

        self._optimizer = optimizer_factory(self._train_model.parameters())
        self._scheduler = SchedulerManager(self._optimizer, lr_scheduler_factory, clip_decay_factory, entropy_decay_factory)
        self._last_model_save_frame = 0
        self._pop_art = PopArt()
        self._first_pop_art_update = True
        self._steps_processor = self._create_steps_processor(None)
        self._target_step = 0

        # self.eval_model_update_interval = 10
        # self._eval_no_copy_updates = 0
        # self.eval_model_blend = 1.0
        # self.eps_nu = 0.1
        # self.eps_alpha = 0.005
        # self.nu = torch.scalar_tensor(1.0, requires_grad=True)
        # self.alpha = torch.scalar_tensor(5.0, requires_grad=True)
        # self._optimizer.add_param_group(dict(params=[self.nu, self.alpha]))

    def _step(self, rewards, dones, states) -> torch.Tensor:
        with torch.no_grad():
            # run network
            ac_out = self._eval_model(states.to(self.device_eval))
            actions = self._eval_model.heads.logits.pd.sample(ac_out.logits).cpu()
            assert not torch.isnan(actions.sum())

            if not self.disable_training:
                ac_out.state_values = ac_out.state_values.squeeze(-1)
                self._steps_processor.append_values(states=states, rewards=rewards, dones=dones, actions=actions, **ac_out)

                if len(self._steps_processor.data.states) > self.horizon:
                    self._train()

            return self.limit_actions(actions)

    def limit_actions(self, actions):
        if isinstance(self.action_space, gym.spaces.Box):
            return actions.clamp(-3, 3) / 3
        else:
            assert isinstance(self.action_space, gym.spaces.Discrete) or \
                   isinstance(self.action_space, gym.spaces.MultiDiscrete)
            return actions

    def _train(self):
        self.step_train = self.step_eval
        self._check_log()
        data = self._create_data()
        self._train_async(data)
        self._scheduler.step(self.frame_train)
        # if self._train_future is not None:
        #     self._train_future.result()
        # self._train_future = self._train_executor.submit(self._train_async, data)

    def _create_data(self):
        self._steps_processor.complete()
        data = self._steps_processor.data
        self._steps_processor = self._create_steps_processor(self._steps_processor)
        return data

    def _train_async(self, data):
        with torch.no_grad():
            log_training_data(self._do_log, self.logger, self.frame_train, self._train_model, data)
            self._ppo_update(data)
            self._model_saver.check_save_model(self._train_model, self.frame_train)

    def _ppo_update(self, data: AttrDict):
        self._apply_pop_art(data)

        data = AttrDict(states=data.states, logits_old=data.logits, state_values_old=data.state_values,
                        actions=data.actions, advantages=data.advantages, value_targets=data.value_targets)

        # if self._has_constraint(Constraint.target):
        #     data.logits_target = self._calc_target_logits(
        #         data.actions, data.logits_old, data.advantages, self._train_model.heads.logits.pd)

        batches = max(1, math.ceil(self.num_actors * self.horizon / self.batch_size))

        # initial_lr = [g['lr'] for g in self._optimizer.param_groups]

        rand_idx = [torch.randperm(len(data.state_values_old), device=self.device_train) for _ in range(self.ppo_iters)]
        rand_idx = torch.cat(rand_idx, 0).chunk(batches * self.ppo_iters)

        old_model = deepcopy(self._train_model.state_dict())
        kl_list = []

        with DataLoader(data, rand_idx, self.device_train, 4) as data_loader:
            for ppo_iter in range(self.ppo_iters):
                for loader_iter in range(batches):
                    # prepare batch data
                    batch = AttrDict(data_loader.get_next_batch())
                    loss, kl = self._ppo_step(batch, self._do_log and ppo_iter == self.ppo_iters - 1 and loader_iter == 0)
                    kl_list.append(kl)

        #         for g in self._optimizer.param_groups:
        #             g['lr'] *= self.lr_iter_mult
        #
        # for g, lr in zip(self._optimizer.param_groups, initial_lr):
        #     g['lr'] = lr

        kl = np.mean(kl_list)

        if self._do_log:
            self.logger.add_scalar('learning_rate', self._learning_rate, self.frame_train)
            self.logger.add_scalar('clip_mult', self._clip_mult, self.frame_train)
            self.logger.add_scalar('total_loss', loss, self.frame_train)
            self.logger.add_scalar('kl', kl, self.frame_train)
            self.logger.add_scalar('kl_scale', self.kl_scale, self.frame_train)
            self.logger.add_scalar('model_abs_diff', model_diff(old_model, self._train_model), self.frame_train)
            self.logger.add_scalar('model_max_diff', model_diff(old_model, self._train_model, True), self.frame_train)

        self._unapply_pop_art()
        # self._adjust_kl_scale(kl)
        # NoisyLinear.randomize_network(self._train_model)

        copy_state_dict(self._train_model, self._eval_model)
        # self._eval_model = deepcopy(self._train_model).to(self.device_eval).eval()

    def _apply_pop_art(self, data):
        if self.use_pop_art:
            self._pop_art.update_statistics(data.value_targets)
            pa_mean, pa_std = self._pop_art.statistics
            if self._first_pop_art_update:
                self._first_pop_art_update = False
            else:
                self._train_model.heads.state_values.normalize(pa_mean, pa_std)
            data.state_values = (data.state_values - pa_mean) / pa_std
            data.value_targets = (data.value_targets - pa_mean) / pa_std
            if self._do_log:
                self.logger.add_scalar('pop_art_mean', pa_mean, self.frame_train)
                self.logger.add_scalar('pop_art_std', pa_std, self.frame_train)

    def _unapply_pop_art(self):
        if self.use_pop_art:
            self._train_model.heads.state_values.unnormalize(*self._pop_art.statistics)

    def _ppo_step(self, batch, do_log):
        with torch.enable_grad():
            actor_params = AttrDict()
            if do_log:
                actor_params.logger = self.logger
                actor_params.cur_step = self.frame_train

            actor_out = self._train_model(batch.states, **actor_params)

            batch.logits = actor_out.logits
            batch.state_values = actor_out.state_values.squeeze(-1)

            for k, v in list(batch.items()):
                batch[k] = v if k == 'states' else v.cpu()

            loss, kl = self._get_ppo_loss(batch, do_log=do_log)
            loss = loss.mean()

        kl = kl.item()
        if (not self._has_constraint(Constraint.spu) or kl < self.spu_dis_agg_lam[1] * self._clip_mult) and \
                (not self._has_constraint(Constraint.kl) or kl < 4 * self.kl_target):
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
        value_targets: Total loss and KL divergence.
        """

        logits, logits_old = batch.logits, batch.logits_old
        values, values_old = batch.state_values, batch.state_values_old
        value_targets = batch.value_targets
        actions = batch.actions
        advantages = batch.advantages

        if pd is None:
            pd = self._train_model.heads.logits.pd

        # clipping factors
        value_clip = self.value_clip * self._clip_mult
        policy_clip = self.policy_clip * self._clip_mult

        # action probability ratio
        # log probabilities used for better numerical stability
        logp_old = pd.logp(actions, logits_old)
        logp = pd.logp(actions, logits)
        ratio = (logp - logp_old).exp()
        kl = pd.kl(logits_old, logits)
        entropy = pd.entropy(logits)

        assert kl.shape == ratio.shape == logp.shape, (kl.shape, ratio.shape, logp.shape)
        assert kl.shape[:-1] == advantages.shape, (kl.shape, advantages.shape)

        loss_ent = -self.entropy_loss_scale * entropy.mean()

        if self._has_constraint(Constraint.clip):
            adv_u = advantages.unsqueeze(-1)
            unclipped_policy_loss = ratio * adv_u
            if self.advantage_scaled_clip:
                pclip = adv_u.abs() * policy_clip
                clipped_ratio = torch.min(torch.max(ratio, 1 - pclip), 1 + pclip)
            else:
                clipped_ratio = ratio.clamp(1 - policy_clip, 1 + policy_clip)
            clipped_policy_loss = clipped_ratio * adv_u
            loss_clip = -torch.min(unclipped_policy_loss, clipped_policy_loss)
            assert loss_clip.shape == logp.shape
        elif self._has_constraint(Constraint.spu):
            kl_dis, _, lam = self.spu_dis_agg_lam
            loss_clip = kl - lam * ratio * advantages.unsqueeze(-1)
            good_kl_mask = kl.detach() <= kl_dis * self._clip_mult
            loss_clip = loss_clip * good_kl_mask.float()
            assert loss_clip.shape == logp.shape
            # agg_kl_check = (kl_mean.detach().mean() < self.kl_agg * self._clip_mult * advantages.abs().mean()).float()
            # loss_clip = loss_clip * agg_kl_check
            # loss_ent = loss_ent * agg_kl_check

        # value loss
        if self._has_constraint(Constraint.clip) or self._has_constraint(Constraint.kl) or self._has_constraint(Constraint.spu):
            if self.advantage_scaled_clip:
                vclip = advantages.abs() * value_clip
                v_pred_clipped = values_old + torch.min(torch.max(values - values_old, -vclip), vclip)
            else:
                v_pred_clipped = values_old + (values - values_old).clamp(-value_clip, value_clip)
            vf_clip_loss = barron_loss(v_pred_clipped, value_targets, *self.barron_alpha_c, reduce=False)
            vf_nonclip_loss = barron_loss(values, value_targets, *self.barron_alpha_c, reduce=False)
            loss_value = self.value_loss_scale * torch.max(vf_nonclip_loss, vf_clip_loss)
            assert loss_value.shape == values.shape
        else:
            loss_value = self.value_loss_scale * barron_loss(values, value_targets, *self.barron_alpha_c, reduce=False)

        # assert loss_clip.shape == loss_value.shape, (loss_clip.shape, loss_value.shape)
        # assert loss_value.shape == loss_ent.shape, (loss_value.shape, loss_ent.shape)
        # assert loss_ent.shape == loss_kl.shape or not self._has_constraint(Constraint.kl), (loss_ent.shape, loss_kl.shape)

        loss_clip = loss_clip.mean()
        loss_ent = loss_ent.mean()
        # loss_kl = loss_kl.mean()
        loss_value = loss_value.mean()

        # sum all losses
        total_loss = loss_clip + loss_value + loss_ent #+ loss_kl
        assert not np.isnan(total_loss.mean().item()) and not np.isinf(total_loss.mean().item()), \
            (loss_clip.mean().item(), loss_value.mean().item(), loss_ent.mean().item())

        if do_log and tag is not None:
            with torch.no_grad():
                self.logger.add_scalar('entropy' + tag, entropy.mean(), self.frame_train)
                self.logger.add_scalar('loss_entropy' + tag, loss_ent.mean(), self.frame_train)
                self.logger.add_scalar('loss_state_value' + tag, loss_value.mean(), self.frame_train)
                self.logger.add_scalar('ratio_mean' + tag, ratio.mean(), self.frame_train)
                self.logger.add_scalar('ratio_abs_mean' + tag, ratio.abs().mean(), self.frame_train)
                self.logger.add_scalar('ratio_abs_max' + tag, ratio.abs().max(), self.frame_train)
                self.logger.add_scalar('loss_policy' + tag, loss_clip.mean(), self.frame_train)

        return total_loss, kl.mean()

    def _has_constraint(self, cons):
        assert isinstance(cons, Constraint)
        return cons in self.constraint

    @property
    def _learning_rate(self):
        return self._optimizer.param_groups[0]['lr']

    @property
    def _clip_mult(self):
        return self._scheduler.clip_decay

    def _log_set(self):
        self.logger.add_text(self.__class__.__name__, pprint.pformat(self._init_args))
        self.logger.add_text('Model', str(self._train_model))

    def drop_collected_steps(self):
        self._steps_processor = self._create_steps_processor(self._steps_processor)

    def _create_steps_processor(self, prev_processor: Optional[StepsProcessor]) -> StepsProcessor:
        return StepsProcessor(self._train_model.heads.logits.pd, self.reward_discount, self.advantage_discount,
                              self.reward_scale, True, self.barron_alpha_c, self.entropy_reward_scale, prev_processor)

    def __getstate__(self):
        d = dict(self.__dict__)
        d['_logger'] = None
        return d

    def __setstate__(self, d):
        self.__dict__ = d
