if __name__ == '__main__':
    from .init_vars import *
    import pybullet_envs

    env_factory = partial(rl.common.SimpleVecEnv, 'AntBulletEnv-v0', parallel='dummy')

    alg_class = rl.algs.PPO
    alg_params = rl.algs.create_fc_kwargs(
        4e6,
        use_pop_art=True,
        constraint='spu',
        ppo_iters=9,
        horizon=256,
        batch_size=256,
        optimizer_factory=partial(optim.Adam, lr=5e-4, eps=1e-5),
        cuda_train=True,
    )
    hparams = dict(
    )
    wrap_params = dict(
        log_root_path=log_path,
        log_interval=10000,
    )

    rl_alg_test(hparams, wrap_params, alg_class, alg_params, env_factory, num_processes=1, iters=1, frames=4e6)
