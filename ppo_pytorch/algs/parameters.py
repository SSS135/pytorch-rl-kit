from functools import partial

import torch.optim as optim

from ..common import DecayLR, ValueDecay
from ..actors import create_ppo_fc_actor, create_ppo_cnn_actor, create_td3_fc_actor


def create_fc_kwargs(learning_decay_frames=None, **kwargs):
    """
    Get hyperparameters for simple envs like CartPole or Acrobot
    Args:
        **kwargs: Any arguments accepted by `PPO`

    Returns: Parameters to initialize PPO
    """

    defaults = dict(
        num_actors=8,
        optimizer_factory=partial(optim.Adam, lr=5e-4),
        policy_clip=0.1,
        value_clip=0.1,
        ppo_iters=6,
        horizon=64,
        batch_size=128,
        model_factory=create_ppo_fc_actor,
        cuda_eval=False,
        cuda_train=False,
    )
    schedulers = dict(
        lr_scheduler_factory=partial(
            DecayLR, start_value=1, end_value=0.01, end_epoch=learning_decay_frames, exp=False),
        clip_decay_factory=partial(
            ValueDecay, start_value=1, end_value=0.01, end_epoch=learning_decay_frames, exp=False),
        entropy_decay_factory=partial(
            ValueDecay, start_value=1, end_value=0.01, end_epoch=learning_decay_frames, exp=True, temp=2),
    ) if learning_decay_frames is not None else dict()
    defaults.update(schedulers)
    defaults.update(kwargs)
    return defaults


def create_td3_fc_kwargs(learning_decay_frames=None, **kwargs):
    """
    Get hyperparameters for simple envs like CartPole or Acrobot
    Args:
        **kwargs: Any arguments accepted by `PPO`

    Returns: Parameters to initialize PPO
    """

    defaults = dict(
    )
    schedulers = dict(
        lr_scheduler_factory=partial(
            DecayLR, start_value=1, end_value=0.01, end_epoch=learning_decay_frames, exp=False),
        entropy_decay_factory=partial(
            ValueDecay, start_value=1, end_value=0.01, end_epoch=learning_decay_frames, exp=True, temp=2),
    ) if learning_decay_frames is not None else dict()
    defaults.update(schedulers)
    defaults.update(kwargs)
    return defaults


def create_atari_kwargs(learning_decay_frames=None, **kwargs):
    """
    Get hyperparameters for Atari
    Args:
        **kwargs: Any arguments accepted by `PPO`

    Returns: Parameters to initialize PPO
    """
    defaults = dict(
        num_actors=8,
        optimizer_factory=partial(optim.Adam, lr=2.5e-4, eps=1e-5),
        horizon=128,
        batch_size=32 * 8,
        constraint='clip',
        value_loss_scale=0.5,
        grad_clip_norm=4,
        ppo_iters=4,
        policy_clip=0.2,
        value_clip=0.2,
        model_factory=create_ppo_cnn_actor,
        cuda_eval=True,
        cuda_train=True,
    )
    schedulers = dict(
        lr_scheduler_factory=partial(
            DecayLR, start_value=1, end_value=0.01, end_epoch=learning_decay_frames, exp=False),
        clip_decay_factory=partial(
            ValueDecay, start_value=1, end_value=0.01, end_epoch=learning_decay_frames, exp=False),
        entropy_decay_factory=partial(
            ValueDecay, start_value=1, end_value=0.01, end_epoch=learning_decay_frames, exp=True, temp=2),
    ) if learning_decay_frames is not None else dict()
    defaults.update(schedulers)
    defaults.update(kwargs)
    return defaults
