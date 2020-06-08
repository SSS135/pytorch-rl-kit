if __name__ == '__main__':
    from .init_vars import *
    from ..algs.parameters import create_ppo_kwargs
    from ppo_pytorch.actors.fc_actors import create_impala_fc_actor
    from ppo_pytorch.common.rl_alg_test import run_training
    from ppo_pytorch.common.variable_env.variable_env_trainer import VariableEnvTrainer
    from ppo_pytorch.common.silu import SiLU
    from ppo_pytorch.algs.impala import IMPALA
    import ppo_pytorch.common.cartpole_continuous
    from ppo_pytorch.common.variable_env.gym_to_variable_env import make_async_env

    train_frames = 2e6
    num_envs = 32
    actors_per_env = 1
    horizon = 64
    env_factory = partial(make_async_env, num_envs=num_envs, env_name='AntBulletEnv-v0', frame_stack=1, frame_skip=1)

    alg_class = IMPALA
    alg_params = create_ppo_kwargs(
        None,

        train_interval_frames=8 * 512,
        train_horizon=horizon,
        batch_size=512,
        value_loss_scale=2.0,
        q_loss_scale=0.0,
        dpg_loss_scale=0.0,
        pg_loss_scale=1.0,
        cuda_eval=False,
        cuda_train=True,

        replay_buf_size=512 * 1024,
        replay_end_sampling_factor=0.05,
        grad_clip_norm=None,
        use_pop_art=False,
        reward_scale=0.1,
        kl_pull=0.05,
        eval_model_blend=0.05,
        vtrace_max_ratio=1.0,
        vtrace_kl_limit=0.3,
        kl_limit=0.3,
        loss_type='impala',
        replay_ratio=7,
        upgo_scale=0.0,
        entropy_loss_scale=0.001,
        barron_alpha_c=(2.0, 1.0),
        memory_burn_in_steps=32,
        activation_norm_scale=0.0,
        reward_reweight_interval=40,
        random_crop_obs=False,

        model_factory=partial(create_impala_fc_actor, hidden_sizes=(64, 64), activation=SiLU),
        # model_factory=partial(rl.actors.create_ppo_rnn_actor, hidden_size=256, num_layers=3),
        optimizer_factory=partial(optim.Adam, lr=5e-4, eps=1e-5),

        # model_init_path='tensorboard\IMPALA_CSBPvP_2020-05-15_12-19-42_[ne16_h128_w-randn_mp]_j15vto_z\model_0.pth',
        # disable_training=True,
    )
    trainer_params = dict(
        rl_alg_factory=partial(alg_class, **alg_params),
        env_factory=env_factory,
        alg_name=alg_class.__name__,
        tag='[h64x2_diag_split_r7]',
        log_root_path=log_path,
        log_interval=10000,
    )

    run_training(VariableEnvTrainer, trainer_params, alg_params, train_frames)
