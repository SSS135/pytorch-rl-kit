from torch_optimizer import Lamb

from ppo_pytorch.common.env_trainer import EnvTrainer
from ppo_pytorch.common.rl_alg_test import run_training
from ppo_pytorch.common.variable_env.gym_to_variable_env import make_async_env
from ppo_pytorch.common.variable_env.variable_env_trainer import VariableEnvTrainer
from ..common.cartpole_continuous import CartPoleContinuousEnv

if __name__ == '__main__':
    from .init_vars import *
    from ..algs.parameters import create_td3_fc_kwargs
    from ppo_pytorch.algs.td3 import TD3
    from ppo_pytorch.common.env_factory import SimpleVecEnv
    import ppo_pytorch.common.cartpole_continuous

    train_frames = 2_000_000
    num_envs = 4
    env_factory = partial(SimpleVecEnv, env_name='Walker2d-v4')

    alg_class = TD3
    alg_params = create_td3_fc_kwargs(
        train_frames,
        cuda_eval=False,
        cuda_train=True,
        num_actors=num_envs,
        train_interval=16,
        batch_size=8 * 1024,
        num_batches=16,
        kl_pull=0.3,
        replay_buffer_size=512*1024,
        replay_end_sampling_factor=1.0,
        reward_scale=0.03,
        rollout_length=16,
        vtrace_kl_limit=2.0,
        actor_update_interval=2,
        entropy_scale=0.0,
        grad_clip_norm=None,
        actor_optimizer_factory=partial(Lamb, lr=0.005),
        critic_optimizer_factory=partial(Lamb, lr=0.005),
    )
    trainer_params = dict(
        rl_alg_factory=partial(alg_class, **alg_params),
        env_factory=env_factory,
        alg_name=alg_class.__name__,
        tag='[w4_kp3_gcnone_kl2stdv]',
        log_root_path=log_path,
        log_interval=2000,
    )

    run_training(EnvTrainer, trainer_params, alg_params, train_frames)
