if __name__ == '__main__':
    from .init_vars import *
    from rl_exp.unity.variable_unity_env import AsyncUnityVecEnv
    from ..common.rl_alg_test import run_training
    from ppo_pytorch.common.variable_env.variable_env_trainer import VariableEnvTrainer
    from ..algs.impala import IMPALA
    from ..algs.parameters import create_ppo_kwargs
    from ..actors.fc_actors import create_impala_fc_actor
    from ..actors.silu import SiLU

    train_frames = 20e6
    num_envs = 8
    actors_per_env = 8 * 1
    visual = False
    horizon = 64
    exe_path = r'c:\Users\Alexander\Projects\DungeonAI\Build\SimpleArenaContinuousVisual\DungeonAI'
    env_factory = partial(AsyncUnityVecEnv, exe_path, num_envs=num_envs, visual_observations=visual, stacked_frames=4,
                          no_graphics_except_first=False, min_ready_envs=0.5)

    alg_class = IMPALA
    alg_params = create_ppo_kwargs(
        train_frames,

        train_interval_frames=horizon * num_envs * actors_per_env,
        train_horizon=horizon,
        batch_size=512,
        value_loss_scale=2.0,
        q_loss_scale=0.0,
        dpg_loss_scale=0.0,
        pg_loss_scale=1.0,
        cuda_eval=True,
        cuda_train=True,

        replay_buf_size=512 * 1024,
        replay_end_sampling_factor=0.05,
        grad_clip_norm=None,
        use_pop_art=False,
        reward_scale=1.0,
        kl_pull=0.05,
        eval_model_blend=0.05,
        vtrace_max_ratio=1.0,
        vtrace_kl_limit=0.3,
        kl_limit=0.3,
        loss_type='impala',
        replay_ratio=3,
        upgo_scale=0.0,
        entropy_loss_scale=0.002,
        barron_alpha_c=(2.0, 1.0),
        memory_burn_in_steps=32,
        activation_norm_scale=0.003,
        num_rewards=3,
        reward_reweight_interval=40,
        random_crop_obs=visual,

        optimizer_factory=partial(optim.AdamW, lr=3e-4, eps=1e-5),
        # model_factory=partial(rl.actors.create_ppo_cnn_actor, cnn_kind='large'),
        # model_factory=partial(rl.actors.create_ppo_rnn_actor, hidden_size=256, num_layers=3),
        model_factory=partial(create_impala_fc_actor, hidden_sizes=(256, 256, 256),
                              activation=SiLU, use_imagination=False),

        # model_init_path=r'c:\Users\Alexander\sync-pc\Jupyter\tensorboard\IMPALA_SimpleArenaContinuous_2020-04-27_15-08-03_[vls1.0_advnorm0.99]_dlwu5k0o\model_0.pth',
        # disable_training=True,
    )
    trainer_params = dict(
        tag='[vt-maxkl_qls0_upgo0_dpg0_mse_pg1_kl0.3_h64_r3_b512_e8]',
        log_root_path=log_path,
        log_interval=20000,
        rl_alg_factory=partial(alg_class, **alg_params),
        env_factory=env_factory,
        alg_name=alg_class.__name__,
    )

    run_training(VariableEnvTrainer, trainer_params, alg_params, train_frames)
