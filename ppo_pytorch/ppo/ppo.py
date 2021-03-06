import math
import pprint
from collections import namedtuple
from functools import partial
from pathlib import Path

import gym.spaces
import numpy as np
import torch
import torch.autograd
import torch.nn.functional as F
import torch.optim as optim
from ..common.opt_clip import opt_clip
from torch.nn.utils import clip_grad_norm_
from torchvision.utils import make_grid

from ..common.barron_loss import barron_loss, barron_loss_derivative
from ..common.gae import calc_advantages, calc_returns
from ..common.probability_distributions import DiagGaussianPd
from ..common.rl_base import RLBase
from ..models import FCActor
from ..models.heads import PolicyHead, StateValueHead


# Used to store env step data for training
Sample = namedtuple('Sample', 'states, rewards, dones, probs, values, actions')
# Preprocessed steps for use in in PPO training loop. Produced from multiple `Sample`.
TrainingData = namedtuple('TrainingData', 'states, probs_old, values_old, actions, advantages, returns, dones, rewards')


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
                 constraint=('kl', 'clip'),
                 policy_clip=0.1,
                 value_clip=0.1,
                 kl_target=0.01,
                 kl_scale=1,
                 cuda_eval=False,
                 cuda_train=False,
                 grad_clip_norm=2,
                 reward_scale=1.0,
                 barron_alpha_c=(1.5, 1),
                 advantage_scaled_clip=True,
                 hidden_code_type: 'input' or 'first' or 'last'='input',
                 image_observation=False,
                 lr_scheduler_factory=None,
                 clip_decay_factory=None,
                 entropy_decay_factory=None,
                 model_save_folder=None,
                 model_save_tag='ppo_model',
                 model_save_interval=None,
                 model_init_path=None,
                 save_intermediate_models=False,
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
            image_observation (bool): Memory optimization for image-based states
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
        self.image_observation = image_observation
        self.model_save_folder = model_save_folder
        self.model_save_interval = model_save_interval
        self.save_intermediate_models = save_intermediate_models
        self.model_save_tag = model_save_tag
        self.kl_target = kl_target
        self.kl_scale = kl_scale
        self.entropy_reward_scale = entropy_reward_scale
        self.hidden_code_type = hidden_code_type
        self.barron_alpha_c = barron_alpha_c
        self.advantage_scaled_clip = advantage_scaled_clip

        assert len(set(self.constraint) - {'clip', 'kl', 'opt'}) == 0
        assert not image_observation or \
               isinstance(observation_space, gym.spaces.Box) and len(observation_space.shape) == 3

        self.sample = self.create_new_sample()

        self.model = model_factory(observation_space, action_space, self.head_factory, hidden_code_type=hidden_code_type)
        if model_init_path is not None:
            self.model.load_state_dict(torch.load(model_init_path))
        self.optimizer = optimizer_factory(self.model.parameters())
        self.lr_scheduler = lr_scheduler_factory(self.optimizer) if lr_scheduler_factory is not None else None
        self.clip_decay = clip_decay_factory() if clip_decay_factory is not None else None
        self.entropy_decay = entropy_decay_factory() if entropy_decay_factory is not None else None
        self.last_model_save_frame = 0
        self.grad_norms = dict()
        # self.value_norm_mean_std = (0, 1)

    def head_factory(self, hidden_size, pd):
        return dict(probs=PolicyHead(hidden_size, pd), state_value=StateValueHead(hidden_size))

    @property
    def learning_rate(self):
        return self.optimizer.param_groups[0]['lr']

    @property
    def clip_mult(self):
        return self.clip_decay.value if self.clip_decay is not None else 1

    def _step(self, prev_states, rewards, dones, cur_states) -> np.ndarray:
        # move network to cuda or cpu
        orig_grad_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(False)

        self.model = self.model.to(self.device_eval)

        # convert observations to tensors
        if self.image_observation:
            states = torch.tensor(cur_states * 255, dtype=torch.uint8)
        else:
            states = torch.tensor(cur_states, dtype=torch.float)

        # run network
        ac_out = self._take_step(states.to(self.device_eval), dones)
        actions = self.model.pd.sample(ac_out.probs).cpu().numpy()

        probs, values = ac_out.probs.cpu().numpy(), ac_out.state_value.cpu().numpy()
        self.append_to_sample(self.sample, states, rewards, dones, actions, probs, values)

        if len(self.sample.rewards) >= self.horizon:
            self._pre_train()
            self._train()

        torch.set_grad_enabled(orig_grad_enabled)

        return actions

    def _take_step(self, states, dones):
        return self.model(states)

    def _pre_train(self):
        self._check_log()

        # update clipping and learning rate decay schedulers
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(self.frame)
        if self.clip_decay is not None:
            self.clip_decay.step(self.frame)
        if self.entropy_decay is not None:
            self.entropy_decay.step(self.frame)

    def _train(self):
        data = self._process_sample(self.sample)
        self._log_training_data(data)
        self._ppo_update(data)
        self.check_save_model()
        self.sample = self.create_new_sample()

    def _log_training_data(self, data):
        if self._do_log:
            if data.states.dim() == 4:
                if data.states.shape[1] in (1, 3):
                    img = data.states[:4]
                    nrow = 2
                else:
                    img = data.states[:4]
                    img = img.view(-1, *img.shape[2:]).unsqueeze(1)
                    nrow = data.states.shape[1]
                if self.image_observation:
                    img = img.float() / 255
                img = make_grid(img, nrow=nrow, normalize=False)
                self.logger.add_image('state', img, self.frame)
            self.logger.add_histogram('rewards', data.rewards, self.frame)
            self.logger.add_histogram('returns', data.returns, self.frame)
            self.logger.add_histogram('advantages', data.advantages, self.frame)
            self.logger.add_histogram('values', data.values_old, self.frame)
            if isinstance(self.model.pd, DiagGaussianPd):
                mean, std = data.probs_old.chunk(2, dim=1)
                self.logger.add_histogram('probs mean', mean, self.frame)
                self.logger.add_histogram('probs std', std, self.frame)
            else:
                self.logger.add_histogram('probs', F.log_softmax(data.probs_old, dim=-1), self.frame)
            for name, param in self.model.named_parameters():
                self.logger.add_histogram(name, param, self.frame)

    @staticmethod
    def append_to_sample(sample, states, rewards, dones, actions, probs, values):
        # add step to history
        if len(sample.states) != 0:
            sample.rewards.append(rewards)
            sample.dones.append(dones)
        sample.states.append(states.cpu())
        sample.probs.append(probs)
        sample.values.append(values)
        sample.actions.append(actions)

    @staticmethod
    def create_new_sample():
        return Sample([], [], [], [], [], [])

    def _process_sample(self, sample, pd=None, reward_discount=None, advantage_discount=None,
                        reward_scale=None, mean_norm=True):
        if pd is None:
            pd = self.model.pd
        if reward_discount is None:
            reward_discount = self.reward_discount
        if advantage_discount is None:
            advantage_discount = self.advantage_discount
        if reward_scale is None:
            reward_scale = self.reward_scale

        # convert list to numpy array
        # (seq, num_actors, ...)
        rewards = torch.tensor(sample.rewards, dtype=torch.float)
        values_old = torch.tensor(sample.values, dtype=torch.float) #* value_norm_std + value_norm_mean
        dones = torch.tensor(sample.dones, dtype=torch.float)
        probs_old = torch.tensor(sample.probs[:-1], dtype=torch.float)

        entropy = pd.entropy(probs_old) * rewards.pow(2).mean().sqrt()
        rewards += self.entropy_reward_scale * entropy

        rewards, returns, advantages = self._process_rewards(
            rewards, values_old, dones, reward_discount, advantage_discount, reward_scale, mean_norm=mean_norm)

        # convert data to Tensors
        actions = torch.tensor(sample.actions[:-1], dtype=pd.dtype)
        values_old = values_old[:-1].reshape(-1)
        states = torch.cat(sample.states[:-1], dim=0) if sample.states is not None else None
        dones = dones.reshape(-1)
        returns = returns.reshape(-1)
        advantages = advantages.reshape(-1)
        rewards = rewards.reshape(-1)

        probs_old, actions = [v.reshape(-1, v.shape[-1]) for v in (probs_old, actions)]

        return TrainingData(states, probs_old, values_old, actions, advantages, returns, dones, rewards)

    def _process_rewards(self, rewards, values, dones, reward_discount, advantage_discount, reward_scale, mean_norm):
        norm_rewards = reward_scale * rewards

        # calculate returns and advantages
        returns = calc_returns(norm_rewards, values, dones, reward_discount)
        advantages = calc_advantages(norm_rewards, values, dones, reward_discount, advantage_discount)
        if mean_norm:
            advantages = (advantages - advantages.mean()) / max(advantages.std(), 1e-3)
        else:
            advantages = advantages / max(advantages.pow(2).mean().sqrt(), 1e-3)
        advantages = barron_loss_derivative(advantages, *self.barron_alpha_c)

        return norm_rewards, returns, advantages

    def _ppo_update(self, data: TrainingData):
        # move model to cuda or cpu
        self.model = self.model.to(self.device_train).train()

        data = (data.states.pin_memory() if self.device_train.type == 'cuda' else data.states,
                data.probs_old, data.values_old, data.actions, data.advantages, data.returns)
        batches = max(1, self.num_actors * self.horizon // self.batch_size)

        for ppo_iter in range(self.ppo_iters):
            rand_idx = torch.randperm(len(data[0])).to(self.device_train)
            for loader_iter in range(batches):
                # prepare batch data
                batch_idx = rand_idx[loader_iter * self.batch_size: (loader_iter + 1) * self.batch_size]
                st, po, vo, ac, adv, ret = [x[batch_idx].to(self.device_train) for x in data]
                if ppo_iter == self.ppo_iters - 1 and loader_iter == 0:
                    self.model.set_log(self.logger, self._do_log, self.step)

                with torch.enable_grad():
                    actor_out = self.model(st)
                    probs = actor_out.probs
                    values = actor_out.state_value
                    # values, vo, ret = [(x - ret_mean) / ret_std for x in (values, vo, ret)]
                    # get loss
                    loss, kl = self._get_ppo_loss(probs, po, values, vo, ac, adv, ret)
                    loss = loss.mean()

                # optimize
                loss.backward()
                if self.grad_clip_norm is not None:
                    clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

                self.model.set_log(self.logger, False, self.step)

            if self._do_log and ppo_iter == self.ppo_iters - 1:
                self.logger.add_scalar('learning rate', self.learning_rate, self.frame)
                self.logger.add_scalar('clip mult', self.clip_mult, self.frame)
                self.logger.add_scalar('total loss', loss, self.frame)
                self.logger.add_scalar('kl', kl, self.frame)

    def _get_ppo_loss(self, probs, probs_old, values, values_old, actions, advantages, returns, pd=None, tag=''):
        """
        Single iteration of PPO algorithm.
        Returns: Total loss and KL divergence.
        """

        if pd is None:
            pd = self.model.pd

        # if tag not in self.grad_norms:
        #     self.grad_norms[tag] = (GradRunningNorm(), GradRunningNorm(self.value_loss_scale))
        # prob_norm, value_norm = self.grad_norms[tag]
        # probs = prob_norm(probs)
        # values = value_norm(values)

        # clipping factors
        value_clip = self.value_clip * self.clip_mult
        policy_clip = self.policy_clip * self.clip_mult

        if 'opt' in self.constraint:
            probs = opt_clip(probs, probs_old, policy_clip)
            values = opt_clip(values, values_old, value_clip)

        # action probability ratio
        # log probabilities used for better numerical stability
        logp_old = pd.logp(actions, probs_old)
        logp = pd.logp(actions, probs)
        ratio = logp - logp_old.detach()

        unclipped_policy_loss = ratio * advantages
        if 'clip' in self.constraint:
            if self.advantage_scaled_clip:
                pclip = advantages.abs() * policy_clip
                clipped_ratio = torch.min(torch.max(ratio, -pclip), pclip)
            else:
                clipped_ratio = ratio.clamp(-policy_clip, policy_clip)
            clipped_policy_loss = clipped_ratio * advantages
            loss_clip = -torch.min(unclipped_policy_loss, clipped_policy_loss)
        else:
            # unclipped loss
            loss_clip = -unclipped_policy_loss

        # value loss
        if self.advantage_scaled_clip:
            vclip = advantages.abs() * value_clip
            v_pred_clipped = values_old + torch.min(torch.max(values - values_old, -vclip), vclip)
        else:
            v_pred_clipped = values_old + (values - values_old).clamp(-value_clip, value_clip)
        vf_clip_loss = barron_loss(v_pred_clipped, returns, *self.barron_alpha_c, reduce=False)
        vf_nonclip_loss = barron_loss(values, returns, *self.barron_alpha_c, reduce=False)
        loss_value = self.value_loss_scale * torch.max(vf_nonclip_loss, vf_clip_loss)

        # entropy bonus for better exploration
        entropy = pd.entropy(probs)
        # entropy_old = pd.entropy(probs_old)

        loss_ent = -self.entropy_loss_scale * entropy
        # loss_ent[(entropy > entropy_old + self.entropy_bonus).detach()] = 0

        kl = pd.kl(probs_old, probs)
        if 'kl' in self.constraint:
            kl_targets = self.kl_target * advantages.abs()
            loss_kl = (kl - kl_targets).div(self.kl_target).pow(2).mul(self.kl_scale * self.kl_target)
            small_kl = (kl < self.kl_target).detach()
            large_kl = (kl > self.kl_target).detach()
            loss_kl[small_kl] = 0
            loss_ent[large_kl] = 0
            loss_clip[large_kl] = 0
        else:
            loss_kl = kl.new(1).zero_()

        assert loss_clip.shape == loss_value.shape
        assert loss_value.shape == loss_ent.shape
        assert loss_ent.shape == loss_kl.shape or 'kl' not in self.constraint
        # sum all losses
        total_loss = loss_clip + loss_value + loss_kl + loss_ent
        assert not np.isnan(total_loss.mean().item()) and not np.isinf(total_loss.mean().item()), \
            (loss_clip.mean().item(), loss_value.mean().item(), loss_ent.mean().item())

        if self.model.do_log and tag is not None:
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

    def _log_set(self):
        self.logger.add_text('PPO', pprint.pformat(self._init_args))
        self.logger.add_text('Model', str(self.model))

    def drop_collected_steps(self):
        self.sample = Sample(states=[], probs=[], values=[], actions=[], rewards=[], dones=[])

    def check_save_model(self):
        if self.model_save_interval is None or \
           self.last_model_save_frame + self.model_save_interval > self.frame:
            return
        self.last_model_save_frame = self.frame
        name = f'{self.model_save_tag}_{self.frame}' if self.save_intermediate_models else self.model_save_tag
        path = Path(self.model_save_folder) / (name + '.pth')
        print('saving to path', path)
        torch.save(self.model.state_dict(), path)
