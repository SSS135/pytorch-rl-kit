if __name__ == '__main__':
    from .init_vars import *
    from optfn.gadam import GAdam

    env_factory = partial(rl.common.SimpleVecEnv, 'Acrobot-v1', parallel='dummy')

    alg_class = rl.algs.IMPALA
    alg_params = rl.algs.create_ppo_kwargs(
        5e5,

        num_actors=8,
        horizon=64,
        batch_size=128,
        model_factory=rl.actors.create_ppo_fc_actor,
        cuda_eval=False,
        cuda_train=False,

        grad_clip_norm=None,
        use_pop_art=True,
        eps_nu_alpha=(1.5, 0.005),
        init_nu_alpha=(1.0, 5.0),
        vtrace_max_ratio=1.0,
        vtrace_kl_limit=0.3,
        loss_type='impala',
        eval_model_update_interval=5,
        replay_ratio=7,
        upgo_scale=0.2,
        # optimizer_factory=partial(optim.Adam, lr=5e-4, eps=1e-5),
        # optimizer_factory=partial(RMSprop, lr=5e-4, eps=0.05),
        optimizer_factory=partial(GAdam, lr=5e-4, betas=(0.9, 0.99), amsgrad_decay=0.0001, eps=1e-4),
    )
    hparams = dict(
    )
    wrap_params = dict(
        log_root_path=log_path,
        log_interval=10000,
    )

    rl_alg_test(hparams, wrap_params, alg_class, alg_params, env_factory, num_processes=1, iters=1, frames=5e5)
