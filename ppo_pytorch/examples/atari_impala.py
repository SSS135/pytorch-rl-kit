if __name__ == '__main__':
    from .init_vars import *
    from optfn.gadam import GAdam
    from torch.optim.adamw import AdamW

    env_factory = partial(rl.common.AtariVecEnv, 'BreakoutNoFrameskip-v4', parallel='process')

    alg_class = rl.algs.IMPALA
    alg_params = rl.algs.create_atari_kwargs(
        10e6,

        num_actors=8,
        horizon=128,
        batch_size=256,
        value_loss_scale=1.0,
        cuda_eval=True,
        cuda_train=True,

        replay_buf_size=256 * 1024,
        replay_end_sampling_factor=0.1,
        grad_clip_norm=None,
        use_pop_art=True,
        kl_pull=0.1,
        vtrace_max_ratio=2.0,
        vtrace_kl_limit=0.1,
        loss_type='impala',
        smooth_model_blend=True,
        eval_model_blend=0.1,
        kl_limit=0.1,
        replay_ratio=7,
        upgo_scale=0.2,
        entropy_loss_scale=1e-3,

        model_factory=partial(rl.actors.create_ppo_cnn_actor, cnn_kind='large'),
        optimizer_factory=partial(AdamW, lr=5e-4, eps=1e-5, weight_decay=1e-5),
    )
    hparams = dict(
    )
    wrap_params = dict(
        tag='[repl7_vklim0.1]',
        log_root_path=log_path,
        log_interval=20000,
    )

    rl_alg_test(hparams, wrap_params, alg_class, alg_params, env_factory, num_processes=1, iters=1, frames=10e6)
