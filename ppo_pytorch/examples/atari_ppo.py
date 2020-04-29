if __name__ == '__main__':
    from .init_vars import *

    env_factory = partial(rl.common.AtariVecEnv, 'BreakoutNoFrameskip-v4', parallel='process')

    alg_class = rl.algs.PPO
    alg_params = rl.algs.create_atari_kwargs(
        10e6,

        num_actors=8,
        horizon=1024 * 10 // 8,
        batch_size=1024,
        value_loss_scale=0.5,
        cuda_eval=True,
        cuda_train=True,

        use_pop_art=False,
        reward_scale=1.0,
        ppo_iters=3,
        constraint='clip',
        value_clip=0.2,
        policy_clip=0.2,
        entropy_loss_scale=0.005,
        grad_clip_norm=None,
        barron_alpha_c=(2.0, 1),
        advantage_scaled_clip=False,

        optimizer_factory=partial(optim.Adam, lr=3e-4),
        model_factory=partial(rl.actors.create_ppo_cnn_actor, cnn_kind='large'),
    )
    hparams = dict(
    )
    wrap_params = dict(
        tag='[newhp]',
        log_root_path=log_path,
        log_interval=20000,
    )

    rl_alg_test(hparams, wrap_params, alg_class, alg_params, env_factory, num_processes=1, iters=1, frames=10e6)
