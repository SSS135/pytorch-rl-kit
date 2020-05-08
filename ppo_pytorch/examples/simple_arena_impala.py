if __name__ == '__main__':
    from .init_vars import *
    from rl_exp.unity.variable_unity_env import VariableUnityVecEnv

    num_envs = 4
    actors_per_env = 8
    exe_path = r'c:\Users\Alexander\Projects\DungeonAI\Build\SimpleArenaContinuous\DungeonAI'
    env_factory = partial(VariableUnityVecEnv, exe_path, num_envs=num_envs, visual_observations=False, stacked_frames=4)
    # env_factory = partial(UnityVecEnv, exe_path, visual_observations=True, stacked_frames=4)

    alg_class = rl.algs.IMPALA
    alg_params = rl.algs.create_ppo_kwargs(
        20e6,

        train_interval_frames=128 * num_envs * actors_per_env,
        train_horizon=128,
        batch_size=512,
        value_loss_scale=1.0,
        q_loss_scale=1.0,
        loss_dpg_scale=0.0,
        cuda_eval=True,
        cuda_train=True,

        # reward_discount=0.997,
        # reward_scale=1.0,

        replay_buf_size=512 * 1024,
        replay_end_sampling_factor=0.05,
        grad_clip_norm=None,
        use_pop_art=False,
        reward_scale=1.0,
        kl_pull=0.05,
        eval_model_blend=0.05,
        vtrace_max_ratio=1.0,
        vtrace_kl_limit=1.0,
        kl_limit=0.2,
        loss_type='impala',
        replay_ratio=7,
        upgo_scale=0.5,
        entropy_loss_scale=0.002,
        barron_alpha_c=(2.0, 1.0),
        memory_burn_in_steps=32,
        activation_norm_scale=0.0,
        num_rewards=3,
        reward_reweight_interval=40,

        optimizer_factory=partial(optim.Adam, lr=3e-4),
        # model_factory=partial(rl.actors.create_ppo_cnn_actor, cnn_kind='large'),
        model_factory=partial(rl.actors.create_ppo_rnn_actor, hidden_size=256, num_layers=3),
        # model_factory=partial(rl.actors.create_ppo_fc_actor, hidden_sizes=(256, 256, 256),
        #                       activation=rl.actors.SiLU),

        # model_init_path=r'c:\Users\Alexander\sync-pc\Jupyter\tensorboard\IMPALA_SimpleArenaContinuous_2020-04-27_15-08-03_[vls1.0_advnorm0.99]_dlwu5k0o\model_0.pth',
        # disable_training=True,
    )
    hparams = dict(
    )
    wrap_params = dict(
        tag='[detsample_oldenc_qls1.0_ord_upgo0.5_vtlim1.0_lrmult_binary_rw_4e8a]',
        log_root_path=log_path,
        log_interval=20000,
    )

    rl_alg_test(hparams, wrap_params, alg_class, alg_params, env_factory, variable_env=True, frames=20e6)
