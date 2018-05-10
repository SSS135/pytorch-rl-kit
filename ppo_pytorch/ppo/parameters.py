from functools import partial

import torch.optim as optim
from ..common import DecayLR, ValueDecay

from ..models import MLPActorCritic, CNNActorCritic


def create_mlp_kwargs(learning_decay_frames=5e5, **kwargs):
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
        model_factory=MLPActorCritic,
        image_observation=False,
        cuda_eval=False,
        cuda_train=False,
        lr_scheduler_factory=partial(
            DecayLR, start_value=1, end_value=0.01, end_epoch=learning_decay_frames, exp=False),
        clip_decay_factory=partial(
            ValueDecay, start_value=1, end_value=0.01, end_epoch=learning_decay_frames, exp=False),
        entropy_decay_factory=partial(
            ValueDecay, start_value=1, end_value=0.01, end_epoch=learning_decay_frames, exp=True, temp=2),
    )
    defaults.update(kwargs)
    return defaults


def create_hqrnn_kwargs(**kwargs):
    """
    Get hyperparameters for simple envs like CartPole or Acrobot
    Args:
        **kwargs: Any arguments accepted by `PPO`

    Returns: Parameters to initialize PPO
    """

    defaults = dict(
        num_actors=16,
        optimizer_factory=partial(optim.Adam, lr=1e-4),
        policy_clip=0.05,
        value_clip=0.1,
        ppo_iters=6,
        horizon=128,
        batch_size=256,
        model_factory=MLPActorCritic,
        image_observation=False,
        cuda_eval=True,
        cuda_train=True,
        lr_scheduler_factory=None,
        clip_decay_factory=None,
        entropy_decay_factory=None,
    )
    defaults.update(kwargs)
    return defaults


def create_atari_kwargs(learning_decay_frames=10e6, **kwargs):
    """
    Get hyperparameters for Atari
    Args:
        **kwargs: Any arguments accepted by `PPO`

    Returns: Parameters to initialize PPO
    """
    defaults = dict(
        num_actors=8,
        optimizer_factory=partial(optim.Adam, lr=2.5e-4, eps=1e-5),
        policy_clip=0.1,
        value_clip=0.1,
        ppo_iters=3,
        constraint='clip_mod',
        grad_clip_norm=3,
        horizon=128,
        batch_size=32 * 8,
        model_factory=CNNActorCritic,
        image_observation=True,
        cuda_eval=True,
        cuda_train=True,
        lr_scheduler_factory=partial(
            DecayLR, start_value=1, end_value=0.01, end_epoch=learning_decay_frames, exp=False),
        clip_decay_factory=partial(
            ValueDecay, start_value=1, end_value=0.01, end_epoch=learning_decay_frames, exp=False),
        entropy_decay_factory=partial(
            ValueDecay, start_value=1, end_value=0.01, end_epoch=learning_decay_frames, exp=True, temp=2),
    )
    defaults.update(kwargs)
    return defaults


def create_sega_kwargs(learning_decay_frames=10e6, **kwargs):
    """
    Get hyperparameters for Atari
    Args:
        **kwargs: Any arguments accepted by `PPO`

    Returns: Parameters to initialize PPO
    """
    defaults = dict(
        num_actors=8,
        optimizer_factory=partial(optim.Adam, lr=2.5e-4, eps=1e-5),
        policy_clip=0.1,
        value_clip=0.1,
        ppo_iters=3,
        constraint='clip_mod',
        grad_clip_norm=3,
        horizon=128,
        batch_size=32 * 8,
        reward_scale=0.0025,
        entropy_bonus=0.001,
        model_factory=None,
        image_observation=True,
        cuda_eval=True,
        cuda_train=True,
        lr_scheduler_factory=None,
        clip_decay_factory=None,
        entropy_decay_factory=None,
    )
    defaults.update(kwargs)
    return defaults
