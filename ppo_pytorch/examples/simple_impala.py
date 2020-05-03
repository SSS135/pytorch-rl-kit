if __name__ == '__main__':
    from .init_vars import *
    from optfn.gadam import GAdam
    from torch.optim.adamw import AdamW

    env_factory = partial(rl.common.SimpleVecEnv, 'BipedalWalkerHardcore-v2', parallel='dummy')

    alg_class = rl.algs.IMPALA
    alg_params = rl.algs.create_ppo_kwargs(
        3e6,

        num_actors=8,
        train_interval_frames=256 * 8,
        train_horizon=128,
        batch_size=256,
        value_loss_scale=1.0,
        cuda_eval=False,
        cuda_train=True,

        # reward_discount=0.997,
        # reward_scale=1.0,

        replay_buf_size=256 * 1024,
        replay_end_sampling_factor=0.1,
        grad_clip_norm=None,
        use_pop_art=True,
        reward_scale=1.0,
        kl_pull=0.1,
        vtrace_max_ratio=1.0,
        vtrace_kl_limit=0.3,
        kl_limit=0.3,
        loss_type='impala',
        eval_model_blend=0.03,
        replay_ratio=7,
        upgo_scale=0.0,
        entropy_loss_scale=0.002,
        barron_alpha_c=(2.0, 1.0),
        memory_burn_in_steps=32,
        activation_norm_scale=0.003,

        model_factory=partial(rl.actors.create_ppo_fc_actor, hidden_sizes=(256, 256, 256),
                              activation=rl.actors.SiLU),
        optimizer_factory=partial(optim.Adam, lr=3e-4),
    )
    hparams = dict(
    )
    wrap_params = dict(
        tag='[r31_pskip1]',
        log_root_path=log_path,
        log_interval=10000,
    )

    rl_alg_test(hparams, wrap_params, alg_class, alg_params, env_factory, num_processes=1, iters=1, frames=3e6)
