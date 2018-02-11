from functools import partial

import torch.optim as optim

from ..models import MLPActorCritic, CNNActorCritic


def create_mlp_kwargs(**kwargs):
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
        learning_decay_frames=5e5,
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
        **kwargs: Any arguments accepted by `PPO`

    Returns: Parameters to initialize PPO
    """
    defaults = dict(
        num_actors=8,
        optimizer_factory=partial(optim.Adam, lr=2.5e-4, eps=1e-5),
        policy_clip=0.1,
        value_clip=0.1,
        ppo_iters=3,
        horizon=128,
        batch_size=32 * 8,
        learning_decay_frames=10e6,
        model_factory=CNNActorCritic,
        image_observation=True,
        cuda_eval=True,
        cuda_train=True,
    )
    defaults.update(kwargs)
    return defaults
