if __name__ == '__main__':
    from .init_vars import *
    import pybullet_envs

    env_factory = partial(rl.common.SimpleVecEnv, 'AntBulletEnv-v0', parallel='dummy')

    alg_class = rl.algs.IMPALA
    alg_params = rl.algs.create_fc_kwargs(
        4e6,
        grad_clip_norm=None,
        use_pop_art=True,
        eps_nu_alpha=(1.5, 0.005),
        init_nu_alpha=(1.0, 5.0),
        vtrace_max_ratio=2.0,
        vtrace_kl_limit=0.5,
        loss_type='impala',
        eval_model_update_interval=5,
        replay_ratio=7,
    )
    hparams = dict(
    )
    wrap_params = dict(
        log_root_path=log_path,
        log_interval=10000,
    )

    rl_alg_test(hparams, wrap_params, alg_class, alg_params, env_factory, num_processes=1, iters=1, frames=4e6)
