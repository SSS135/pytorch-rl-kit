if __name__ == '__main__':
    from .init_vars import *
    from optfn.gadam import GAdam
    from torch.optim.adamw import AdamW

    env_factory = partial(rl.common.AtariVecEnv, 'BreakoutNoFrameskip-v4', parallel='process')

    alg_class = rl.algs.IMPALA
    alg_params = rl.algs.create_ppo_kwargs(
        10e6,

        num_actors=8,
        horizon=128,
        batch_size=256,
        value_loss_scale=0.5,
        cuda_eval=True,
        cuda_train=True,

        # reward_discount=0.997,
        # reward_scale=1.0,

        replay_buf_size=256 * 1024,
        replay_end_sampling_factor=0.1,
        grad_clip_norm=None,
        use_pop_art=True,
        reward_scale=1.0,
        kl_pull=0.1,
        eval_model_blend=0.03,
        vtrace_max_ratio=1.0,
        vtrace_kl_limit=0.2,
        kl_limit=0.2,
        loss_type='impala',
        replay_ratio=7,
        upgo_scale=0.0,
        entropy_loss_scale=0.005,
        barron_alpha_c=(2.0, 1.0),

        optimizer_factory=partial(optim.Adam, lr=3e-4),

        model_factory=partial(rl.actors.create_ppo_cnn_actor, cnn_kind='large'),
    )
    hparams = dict(
    )
    wrap_params = dict(
        tag='[popart_vls0.5_rscale]',
        log_root_path=log_path,
        log_interval=20000,
    )

    rl_alg_test(hparams, wrap_params, alg_class, alg_params, env_factory, num_processes=1, iters=1, frames=10e6)
