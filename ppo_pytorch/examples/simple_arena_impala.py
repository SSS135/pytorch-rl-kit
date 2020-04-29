if __name__ == '__main__':
    from .init_vars import *
    from rl_exp.unity.simple_unity_env import UnityVecEnv

    exe_path = r'c:\Users\Alexander\Projects\DungeonAI\Build\SimpleArenaContinuous\DungeonAI'
    env_factory = partial(UnityVecEnv, exe_path)
    # env_factory = partial(UnityVecEnv, exe_path, visual_observations=True, stacked_frames=4)

    alg_class = rl.algs.IMPALA
    alg_params = rl.algs.create_ppo_kwargs(
        20e6,

        num_actors=8,
        horizon=128,
        train_horizon=64,
        batch_size=64,
        value_loss_scale=1.0,
        cuda_eval=False,
        cuda_train=True,

        # reward_discount=0.997,
        # reward_scale=1.0,

        replay_buf_size=256 * 1024,
        replay_end_sampling_factor=0.1,
        grad_clip_norm=None,
        use_pop_art=False,
        reward_scale=1.0,
        kl_pull=0.1,
        vtrace_max_ratio=1.0,
        vtrace_kl_limit=0.2,
        kl_limit=0.2,
        loss_type='impala',
        eval_model_blend=0.1,
        replay_ratio=7,
        upgo_scale=0.0,
        entropy_loss_scale=0.005,
        barron_alpha_c=(2.0, 1.0),
        value_pull=0.0,
        memory_burn_in_steps=32,

        optimizer_factory=partial(optim.Adam, lr=3e-4, eps=1e-5),
        # model_factory=partial(rl.actors.create_ppo_cnn_actor, cnn_kind='normal'),
        model_factory=partial(rl.actors.create_ppo_rnn_actor, hidden_size=256, num_layers=3),
        # model_factory=partial(rl.actors.create_ppo_fc_actor, hidden_sizes=(256, 256, 256),
        #                       activation=rl.actors.SiLU),

        # model_init_path=r'c:\Users\Alexander\sync-pc\Jupyter\tensorboard\IMPALA_SimpleArenaContinuous_2020-04-27_15-08-03_[vls1.0_advnorm0.99]_dlwu5k0o\model_0.pth',
        # disable_training=True,
    )
    hparams = dict(
    )
    wrap_params = dict(
        tag='[rnn_diagmeankl_bs64_eps1e-5]',
        log_root_path=log_path,
        log_interval=20000,
    )

    rl_alg_test(hparams, wrap_params, alg_class, alg_params, env_factory, num_processes=1, iters=1, frames=20e6)
