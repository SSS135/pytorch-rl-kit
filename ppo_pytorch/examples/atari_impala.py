if __name__ == '__main__':
    from .init_vars import *
    from optfn.gadam import GAdam

    env_factory = partial(rl.common.AtariVecEnv, 'BreakoutNoFrameskip-v4', parallel='process')

    alg_class = rl.algs.IMPALA
    alg_params = rl.algs.create_atari_kwargs(
        10e6,
        grad_clip_norm=None,
        use_pop_art=True,
        eps_nu_alpha=(0.05, 0.005),
        init_nu_alpha=(1.0, 5.0),
        vtrace_max_ratio=1.0,
        vtrace_kl_limit=0.5,
        loss_type='v_mpo',
        eval_model_update_interval=5,
        replay_ratio=7,
        model_factory=partial(rl.actors.create_ppo_cnn_actor, cnn_kind='large'),
        upgo_scale=0.2,
        # optimizer_factory=partial(RMSprop, lr=5e-4, eps=0.05),
        # optimizer_factory=partial(GAdam, lr=2.5e-4, betas=(0.9, 0.99), amsgrad_decay=0.0001, eps=1e-4),
        optimizer_factory=partial(optim.Adam, lr=2.5e-4, eps=1e-5),
    )
    hparams = dict(
    )
    wrap_params = dict(
        tag='[vmpo_const_clip_0.05]',
        log_root_path=log_path,
        log_interval=20000,
    )

    rl_alg_test(hparams, wrap_params, alg_class, alg_params, env_factory, num_processes=1, iters=1, frames=10e6)
