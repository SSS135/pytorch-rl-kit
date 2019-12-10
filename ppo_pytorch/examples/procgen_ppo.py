if __name__ == '__main__':
    from .init_vars import *
    from optfn.gadam import GAdam
    import procgen

    env_factory = partial(rl.common.ProcgenVecEnv, 'procgen:procgen-caveflyer-easy-v0', parallel='thread')

    alg_class = rl.algs.PPO
    alg_params = rl.algs.create_fc_kwargs(
        20e6,

        use_pop_art=False,
        num_actors=16,
        horizon=128,
        batch_size=256,
        constraint='clip',
        value_loss_scale=0.5,
        grad_clip_norm=0.5,
        reward_discount=0.999,
        ppo_iters=3,
        policy_clip=0.2,
        value_clip=0.2,
        model_factory=partial(rl.actors.create_ppo_cnn_actor, cnn_kind='large'),
        cuda_eval=True,
        cuda_train=True,
        # upgo_scale=0.0,
        optimizer_factory=partial(optim.Adam, lr=5e-4, eps=1e-5),
        # optimizer_factory=partial(GAdam, lr=3e-4, betas=(0.9, 0.9), amsgrad_decay=0.01),
    )
    hparams = dict(
    )
    wrap_params = dict(
        tag='[kl0.5]',
        log_root_path=log_path,
        log_interval=20000,
    )

    rl_alg_test(hparams, wrap_params, alg_class, alg_params, env_factory, num_processes=1, iters=1, frames=20e6)
