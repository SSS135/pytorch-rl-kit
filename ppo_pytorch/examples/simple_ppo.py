from ppo_pytorch.algs.parameters import create_fc_kwargs
from ppo_pytorch.algs.ppo import PPO
from ppo_pytorch.common.env_factory import SimpleVecEnv

if __name__ == '__main__':
    from .init_vars import *

    env_factory = partial(SimpleVecEnv, 'CartPole-v1', parallel='dummy')

    alg_class = PPO
    alg_params = create_fc_kwargs(
        5e5,

        num_actors=8,
        horizon=128,
        batch_size=256,
        value_loss_scale=0.5,
        cuda_eval=False,
        cuda_train=True,
        batch_kl_limit=None,

        use_pop_art=False,
        reward_scale=0.1,
        ppo_iters=6,
        # constraint='clip',
        value_clip=0.2,
        policy_clip=0.2,
        entropy_loss_scale=0.005,
        grad_clip_norm=2,
        target_model_blend=1,
        # barron_alpha_c=(2.0, 1),
        # advantage_scaled_clip=False,

        optimizer_factory=partial(optim.Adam, lr=3e-4),
    )
    hparams = dict(
    )
    wrap_params = dict(
        tag='[noblend_localadv]',
        log_root_path=log_path,
        log_interval=10000,
    )

    rl_alg_test(hparams, wrap_params, alg_class, alg_params, env_factory, num_processes=1, iters=1, frames=5e5)
