from torch_optimizer import Lamb, Adahessian

from ppo_pytorch.algs.parameters import create_fc_kwargs
from ppo_pytorch.algs.ppo import PPO
from ppo_pytorch.common.env_factory import SimpleVecEnv
from ppo_pytorch.common.env_trainer import EnvTrainer
from ppo_pytorch.common.rl_alg_test import run_training
from ..common.cartpole_continuous import CartPoleContinuousEnv

if __name__ == '__main__':
    from .init_vars import *

    train_frames = 2_000_000
    num_envs = 64
    env_factory = partial(SimpleVecEnv, env_name='CartPole-v1')

    alg_class = PPO
    alg_params = create_fc_kwargs(
        num_actors=num_envs,
        horizon=128,
        batch_size=8 * 1024,
        cuda_eval=True,
        cuda_train=True,

        kl_pull=0.5,
        use_pop_art=True,
        reward_scale=1.0,
        ppo_iters=15,
        value_clip=None,
        policy_clip=0.3,
        entropy_loss_scale=0.0,
        grad_clip_norm=None,
        target_model_blend=1,
        batch_kl_limit=0.03,

        optimizer_factory=partial(Adahessian, lr=0.5),
    )
    trainer_params = dict(
        rl_alg_factory=partial(alg_class, **alg_params),
        env_factory=env_factory,
        alg_name=alg_class.__name__,
        tag='[ah_lr5]',
        log_root_path=log_path,
        log_interval=10000,
    )

    run_training(EnvTrainer, trainer_params, alg_params, train_frames)
