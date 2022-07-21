import math
import pprint
from asyncio import Future
from concurrent.futures.thread import ThreadPoolExecutor
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
from ppo_pytorch.common.activation_norm import activation_norm_loss
from rl_exp.noisy_linear import NoisyLinear
from torch.nn.utils import clip_grad_norm_
from torchvision.utils import make_grid

from .steps_processor import StepsProcessor
from .utils import lerp_module_
from ..actors.fc_actors import create_ppo_fc_actor
from ..actors.actors import Actor
from ..actors.utils import model_diff
from ..common.attr_dict import AttrDict
from ..common.barron_loss import barron_loss
from ..common.data_loader import DataLoader
from ..common.pop_art import PopArt
from ..common.probability_distributions import DiagGaussianPd, CategoricalPd, ProbabilityDistribution
from ..common.rl_base import RLBase, RLStepData
    
    
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
    if not do_log:
        return

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
    targets = data.state_value_targets
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
                 policy_clip=0.2,
                 value_clip: Optional[float] = 0.2,
                 kl_pull=0.0,
                 batch_kl_limit=1.0,
                 cuda_eval=False,
                 cuda_train=False,
                 grad_clip_norm=2,
                 reward_scale=1.0,
                 lr_scheduler_factory=None,
                 clip_decay_factory=None,
                 entropy_decay_factory=None,
                 use_pop_art=False,
                 activation_norm_scale=0.0,
                 squash_values=False,
                 target_model_blend=0.1,
                 **kwargs):
        """
        Single threaded implementation of Proximal Policy Optimization Algorithms
        https://arxiv.org/pdf/1707.06347.pdf

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
            policy_clip (float): policy clip strength
            value_clip (float): State-value clip strength
            cuda_eval (bool): Use CUDA for environment steps
            cuda_train (bool): Use CUDA for training steps
            grad_clip_norm (float or None): Max norm for gradient clipping (typically 0.5 to 40)
            reward_scale (float): Scale factor for environment's rewards
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
        self.kl_pull = kl_pull
        self.batch_kl_limit = batch_kl_limit
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
        self.use_pop_art = use_pop_art
        self.lr_scheduler_factory = lr_scheduler_factory
        self.activation_norm_scale = activation_norm_scale
        self.squash_values = squash_values
        self.target_model_blend = target_model_blend

        self._train_model: Actor = model_factory(observation_space, action_space)
        self._target_model: Actor = model_factory(observation_space, action_space)
        self._eval_model: Actor = model_factory(observation_space, action_space)
        if self.model_init_path is not None:
            self._train_model.load_state_dict(torch.load(self.model_init_path), True)
            print(f'loaded model {self.model_init_path}')
        copy_state_dict(self._train_model, self._target_model)
        copy_state_dict(self._train_model, self._eval_model)
        self._train_model = self._train_model.train().to(self.device_train, non_blocking=True)
        self._target_model = self._target_model.train().to(self.device_train, non_blocking=True)
        self._eval_model = self._eval_model.eval().to(self.device_eval, non_blocking=True)

        self._optimizer = optimizer_factory(self._train_model.parameters())
        self._scheduler = SchedulerManager(self._optimizer, lr_scheduler_factory, clip_decay_factory, entropy_decay_factory)
        self._last_model_save_frame = 0
        self._pop_art = PopArt()
        self._first_pop_art_update = True
        self._steps_processor = self._create_steps_processor()
        self._target_step = 0

        self._train_future: Optional[Future] = None
        self._train_executor = ThreadPoolExecutor(max_workers=1)

    def _step(self, data: RLStepData) -> torch.Tensor:
        with torch.no_grad():
            # run network
            ac_out = self._eval_model(data.obs.to(self.device_eval))
            actions = self._eval_model.heads.logits.pd.sample(ac_out.logits).cpu()
            assert not torch.isnan(actions.sum())

            if not self.disable_training:
                ac_out.state_values = ac_out.state_values.squeeze(-1)
                self._steps_processor.append_values(states=data.obs, rewards=data.rewards.sum(-1),
                                                    dones=data.done, actions=actions, **ac_out)

                if len(self._steps_processor.data.states) > self.horizon:
                    self._train()

            return self._eval_model.heads.logits.pd.postprocess_action(actions)

    def _train(self):
        with torch.no_grad():
            if self._train_future is not None:
                self._train_future.result()

            self.frame_train = self.frame_eval
            self._check_log()

            old_sp = self._steps_processor
            self._steps_processor = self._create_steps_processor()
            self._train_future = self._train_executor.submit(self._train_async, old_sp)

    def _create_data(self):
        self._steps_processor.complete()
        data = self._steps_processor.data
        self._steps_processor = self._create_steps_processor()
        return data

    def _train_async(self, steps_processor):
        with torch.no_grad():
            steps_processor.complete()
            data = steps_processor.data

            log_training_data(self._do_log, self.logger, self.frame_train, self._train_model, data)
            self._ppo_update(data)
            self._model_saver.check_save_model(self._train_model, self.frame_train)
            self._scheduler.step(self.frame_train)

    def _ppo_update(self, data: AttrDict):
        self._apply_pop_art(data)

        data = AttrDict(states=data.states, logits_old=data.logits, state_values_old=data.state_values,
                        actions=data.actions, advantages=data.advantages, state_value_targets=data.state_value_targets)

        batches = max(1, math.ceil(self.num_actors * self.horizon / self.batch_size))

        rand_idx = [torch.randperm(len(data.state_values_old), device=self.device_train) for _ in range(self.ppo_iters)]
        rand_idx = torch.cat(rand_idx, 0).chunk(batches * self.ppo_iters)

        old_model = deepcopy(self._train_model.state_dict())
        kl_list = []

        with DataLoader(data, rand_idx, self.device_train, 4) as data_loader:
            for ppo_iter in range(self.ppo_iters):
                for loader_iter in range(batches):
                    batch = AttrDict(data_loader.get_next_batch())
                    loss, kl = self._ppo_step(batch, self._do_log and ppo_iter == self.ppo_iters - 1 and loader_iter == 0)
                    kl_list.append(kl)

        kl = np.mean(kl_list)

        if self._do_log:
            self.logger.add_scalar('Optimizer/Learning Rate', self._learning_rate, self.frame_train)
            self.logger.add_scalar('Optimizer/Clip Mult', self._clip_mult, self.frame_train)
            self.logger.add_scalar('Losses/Total Loss', loss, self.frame_train)
            self.logger.add_scalar('Stability/KL Blend', kl, self.frame_train)
            self.logger.add_scalar('Model Diff/Abs', model_diff(old_model, self._train_model), self.frame_train)
            self.logger.add_scalar('Model Diff/Max', model_diff(old_model, self._train_model, True), self.frame_train)

        self._unapply_pop_art()
        # NoisyLinear.randomize_network(self._train_model)

        copy_state_dict(self._train_model, self._eval_model)
        lerp_module_(self._target_model, self._train_model, self.target_model_blend)

    def _apply_pop_art(self, data):
        if self.use_pop_art:
            self._pop_art.update_statistics(data.state_value_targets)
            pa_mean, pa_std = self._pop_art.statistics
            if self._first_pop_art_update:
                self._first_pop_art_update = False
            else:
                self._train_model.heads.state_values.normalize(pa_mean, pa_std)
            data.state_values = (data.state_values - pa_mean) / pa_std
            data.state_value_targets = (data.state_value_targets - pa_mean) / pa_std
            if self._do_log:
                self.logger.add_scalar('PopArt/Mean', pa_mean, self.frame_train)
                self.logger.add_scalar('PopArt/Std', pa_std, self.frame_train)

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
            if self.target_model_blend < 1:
                with torch.no_grad():
                    actor_out_target = self._target_model(batch.states, evaluate_heads=['logits'], **actor_params)

            batch.logits = actor_out.logits
            batch.logits_target = actor_out_target.logits if self.target_model_blend < 1 else batch.logits_old
            batch.state_values = actor_out.state_values.squeeze(-1)

            for k, v in list(batch.items()):
                batch[k] = v if k == 'states' else v.cpu()

            loss, kl = self._get_ppo_loss(batch, do_log=do_log)
            act_norm_loss = activation_norm_loss(self._train_model).cpu()
            loss = loss.mean() + self.activation_norm_scale * act_norm_loss

        kl = kl.item()
        if self.batch_kl_limit is None or kl < self.batch_kl_limit * self._clip_mult:
            # optimize
            loss.backward()
            if self.grad_clip_norm is not None:
                clip_grad_norm_(self._train_model.parameters(), self.grad_clip_norm)
            self._optimizer.step()
            self._optimizer.zero_grad()

        if do_log:
            self.logger.add_scalar('Losses/Activation Norm', act_norm_loss, self.frame_train)

        return loss, kl

    def _get_ppo_loss(self, batch, pd=None, do_log=False, tag=''):
        logits, logits_old, logits_target = batch.logits, batch.logits_old, batch.logits_target
        values, values_old = batch.state_values, batch.state_values_old
        state_value_targets = batch.state_value_targets
        actions = batch.actions
        advantages = batch.advantages

        if pd is None:
            pd = self._train_model.heads.logits.pd

        logp_target = pd.logp(actions, logits_target)
        logp = pd.logp(actions, logits)
        # ratio = (logp - logp_old).mean(-1)
        ratio = (logp - logp_target).mean(-1)
        kl = pd.kl(logits_target, logits).mean(-1)
        entropy = pd.entropy(logits)

        assert kl.shape == ratio.shape == logp.shape[:-1], (kl.shape, ratio.shape, logp.shape)
        assert kl.shape == advantages.shape, (kl.shape, advantages.shape)

        loss_ent = -self.entropy_loss_scale * entropy.mean()
        # loss_kl = self.kl_pull * kl.mean()
        loss_kl = (logits - logits_target).pow_(2).mean().mul(0.5 * self.kl_pull)

        policy_clip = self.policy_clip * self._clip_mult
        unclipped_policy_loss = ratio * advantages
        clipped_policy_loss = ratio.clamp(-policy_clip, policy_clip) * advantages
        loss_clip = -torch.min(unclipped_policy_loss, clipped_policy_loss)
        assert loss_clip.shape == advantages.shape

        # value loss
        if self.value_clip is not None:
            value_clip = self.value_clip * self._clip_mult
            v_pred_clipped = values_old + (values - values_old).clamp_(-value_clip, value_clip)
            vf_clip_loss = (v_pred_clipped - state_value_targets).pow_(2).mul_(0.5)
            vf_nonclip_loss = (values - state_value_targets).pow_(2).mul_(0.5)
            loss_value = self.value_loss_scale * torch.max(vf_nonclip_loss, vf_clip_loss)
            assert loss_value.shape == values.shape
        else:
            loss_value = self.value_loss_scale * (values - state_value_targets).pow_(2).mul_(0.5)

        # assert loss_clip.shape == loss_value.shape, (loss_clip.shape, loss_value.shape)
        # assert loss_value.shape == loss_ent.shape, (loss_value.shape, loss_ent.shape)

        loss_clip = loss_clip.mean()
        loss_value = loss_value.mean()

        # sum all losses
        total_loss = loss_clip + loss_value + loss_ent + loss_kl
        assert not np.isnan(total_loss.mean().item()) and not np.isinf(total_loss.mean().item()), \
            (loss_clip.mean().item(), loss_value.mean().item(), loss_ent.mean().item())

        if do_log and tag is not None:
            with torch.no_grad():
                self.logger.add_scalar('Stability/Entropy' + tag, entropy.mean(), self.frame_train)
                self.logger.add_scalar('Losses/State Value' + tag, loss_value.mean(), self.frame_train)
                self.logger.add_scalar('Prob Ratio/Mean' + tag, ratio.mean(), self.frame_train)
                self.logger.add_scalar('Prob Ratio/Abs Mean' + tag, ratio.abs().mean(), self.frame_train)
                self.logger.add_scalar('Prob Ratio/Abs Max' + tag, ratio.abs().max(), self.frame_train)

        return total_loss, kl.mean()

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
        self._steps_processor = self._create_steps_processor()

    def _create_steps_processor(self) -> StepsProcessor:
        return StepsProcessor(self._train_model.heads.logits.pd, self.reward_discount, self.advantage_discount,
                              self.reward_scale, True, self.squash_values)

    def __getstate__(self):
        d = dict(self.__dict__)
        d['_logger'] = None
        return d

    def __setstate__(self, d):
        self.__dict__ = d
