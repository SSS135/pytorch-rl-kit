from torch_optimizer import Lamb, Adahessian

from ppo_pytorch.algs.advppo import AdvPPO
from ppo_pytorch.algs.parameters import create_fc_kwargs
from ppo_pytorch.algs.ppo import PPO
from ppo_pytorch.common.env_factory import SimpleVecEnv
from ppo_pytorch.common.env_trainer import EnvTrainer
from ppo_pytorch.common.rl_alg_test import run_training
from ..common.cartpole_continuous import CartPoleContinuousEnv

if __name__ == '__main__':
    from .init_vars import *

    train_frames = 2_000_000
    num_envs = 32
    env_factory = partial(SimpleVecEnv, env_name='CartPoleContinuous-v1')

    alg_class = AdvPPO
    alg_params = dict(
        num_actors=num_envs,
        horizon=128,
        batch_size=4 * 1024,
        cuda_eval=True,
        cuda_train=True,

        use_pop_art=False,
        reward_scale=0.03,
        ppo_iters=4,
        value_clip=None,
        grad_clip_norm=None,
        random_policy_frames=128 * 1024,
        gan_queue_len=16,

        optimizer_factory=partial(Lamb, lr=0.005),
    )
    trainer_params = dict(
        rl_alg_factory=partial(alg_class, **alg_params),
        env_factory=env_factory,
        alg_name=alg_class.__name__,
        tag='[]',
        log_root_path=log_path,
        log_interval=10000,
    )

    run_training(EnvTrainer, trainer_params, alg_params, train_frames)
