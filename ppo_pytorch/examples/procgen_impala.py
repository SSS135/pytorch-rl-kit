if __name__ == '__main__':
    from .init_vars import *
    from optfn.gadam import GAdam
    import procgen

    env_factory = partial(rl.common.ProcgenVecEnv, 'procgen:procgen-coinrun-v0', parallel='process')

    alg_class = rl.algs.IMPALA
    alg_params = rl.algs.create_ppo_kwargs(
        10e6,

        num_actors=8,
        horizon=128,
        batch_size=32 * 8,
        value_loss_scale=0.5,
        cuda_eval=True,
        cuda_train=True,

        grad_clip_norm=None,
        use_pop_art=True,
        eps_nu_alpha=(0.76, 0.005),
        init_nu_alpha=(1.0, 5.0),
        vtrace_max_ratio=1.0,
        vtrace_kl_limit=0.2,
        loss_type='impala',
        eval_model_update_interval=5,
        replay_ratio=7,
        upgo_scale=0.2,
        model_factory=partial(rl.actors.create_ppo_cnn_actor, cnn_kind='large'),
        # optimizer_factory=partial(optim.Adam, lr=3e-4, eps=1e-5),
        # optimizer_factory=partial(GAdam, lr=3e-4, betas=(0.9, 0.9), amsgrad_decay=0.01),
        optimizer_factory=partial(GAdam, lr=5e-4, betas=(0.9, 0.99), amsgrad_decay=0.0001, eps=1e-4),
    )
    hparams = dict(
    )
    wrap_params = dict(
        log_root_path=log_path,
        log_interval=20000,
    )

    rl_alg_test(hparams, wrap_params, alg_class, alg_params, env_factory, num_processes=1, iters=1, frames=10e6)
