from functools import partial

import torch.optim as optim

from ..models import MLPActorCritic, CNNActorCritic


def create_mlp_kwargs(**kwargs):
    """
    Get hyperparameters for simple envs like CartPole or Acrobot
    Args:
        train_frames: Frames to decay learning rate and clipping.
        **kwargs: Any arguments accepted by `PPO`

    Returns: Parameters to initialize PPO
    """
    defaults = dict(
        num_actors=8,
        optimizer_factory=partial(optim.Adam, lr=5e-4),
        policy_clip=0.1,
        value_clip=0.1,
        ppo_iters=6,
        entropy_bonus=0.01,
        horizon=64,
        batch_size=128,
        grad_clip_norm=0.5,
        value_loss_scale=0.5,
        learning_decay_frames=2e5,
        model_factory=MLPActorCritic,
        image_observation=False,
        cuda_eval=False,
        cuda_train=False,
    )
    defaults.update(kwargs)
    return defaults


def create_cnn_kwargs(**kwargs):
    """
    Get hyperparameters for Atari
    Args:
        train_frames: Frames to decay learning rate and clipping.
        **kwargs: Any arguments accepted by `PPO`

    Returns: Parameters to initialize PPO
    """
    defaults = dict(
        num_actors=8,
        optimizer_factory=partial(optim.Adam, lr=2.5e-4, eps=1e-5),
        policy_clip=0.1,
        value_clip=0.1,
        ppo_iters=4,
        entropy_bonus=0.01,
        horizon=128,
        batch_size=32 * 8,
        value_loss_scale=0.5,
        grad_clip_norm=0.5,
        learning_decay_frames=10e6,
        model_factory=partial(CNNActorCritic, cnn_kind='large'),
        image_observation=True,
        cuda_eval=True,
        cuda_train=True,
    )
    defaults.update(kwargs)
    return defaults
