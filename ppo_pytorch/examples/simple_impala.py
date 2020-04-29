if __name__ == '__main__':
    from .init_vars import *
    from optfn.gadam import GAdam
    from torch.optim.adamw import AdamW

    env_factory = partial(rl.common.SimpleVecEnv, 'CartPoleContinuous-v1', parallel='dummy')

    alg_class = rl.algs.IMPALA
    alg_params = rl.algs.create_ppo_kwargs(
        5e5,

        num_actors=8,
        horizon=256,
        batch_size=512,
        value_loss_scale=0.5,
        cuda_eval=False,
        cuda_train=True,

        # reward_discount=0.997,
        # reward_scale=1.0,

        replay_buf_size=256 * 1024,
        replay_end_sampling_factor=1.0,
        grad_clip_norm=None,
        use_pop_art=False,
        reward_scale=0.05,
        kl_pull=0.1,
        vtrace_max_ratio=1.0,
        vtrace_kl_limit=0.5,
        kl_limit=0.5,
        loss_type='impala',
        eval_model_blend=0.1,
        replay_ratio=7,
        upgo_scale=0.0,
        entropy_loss_scale=0.005,
        barron_alpha_c=(2.0, 1.0),

        model_factory=partial(rl.actors.create_ppo_fc_actor, hidden_sizes=(128, 128),
                              activation=rl.actors.SiLU),
        optimizer_factory=partial(optim.Adam, lr=3e-4),
    )
    hparams = dict(
    )
    wrap_params = dict(
        tag='[kl0.5_s1.0_advnorm-m0.9_h256_b512_rs0.05]',
        log_root_path=log_path,
        log_interval=10000,
    )

    rl_alg_test(hparams, wrap_params, alg_class, alg_params, env_factory, num_processes=1, iters=1, frames=5e5)
