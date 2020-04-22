if __name__ == '__main__':
    from .init_vars import *
    from optfn.gadam import GAdam

    env_factory = partial(rl.common.SimpleVecEnv, 'CartPoleContinuous-v1', parallel='dummy')

    alg_class = rl.algs.IMPALA
    alg_params = rl.algs.create_ppo_kwargs(
        5e5,

        num_actors=8,
        horizon=64,
        batch_size=128,
        model_factory=rl.actors.create_ppo_fc_actor,
        cuda_eval=False,
        cuda_train=False,

        replay_buf_size=128 * 1024,
        replay_end_sampling_factor=0.01,
        grad_clip_norm=None,
        use_pop_art=True,
        kl_pull=0.1,
        vtrace_max_ratio=2.0,
        vtrace_kl_limit=0.5,
        loss_type='impala',
        eval_model_blend=0.1,
        kl_limit=0.01,
        replay_ratio=3,
        upgo_scale=0.2,
        entropy_loss_scale=1e-3,

        optimizer_factory=partial(optim.Adam, lr=5e-4, eps=1e-5),
        # optimizer_factory=partial(RMSprop, lr=5e-4, eps=0.05),
        # optimizer_factory=partial(GAdam, lr=5e-4, betas=(0.9, 0.99), amsgrad_decay=0.0001, eps=1e-4),
    )
    hparams = dict(
    )
    wrap_params = dict(
        tag='[64ac]',
        log_root_path=log_path,
        log_interval=10000,
    )

    rl_alg_test(hparams, wrap_params, alg_class, alg_params, env_factory, num_processes=1, iters=1, frames=5e5)
