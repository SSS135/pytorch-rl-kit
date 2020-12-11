

if __name__ == '__main__':
    from .init_vars import *
    from raic2020 import make_env
    from ..common.rl_alg_test import run_training
    from ..common.variable_env.variable_self_play_trainer import VariableSelfPlayTrainer
    from ..common.variable_env.variable_env_trainer import VariableEnvTrainer
    from ..algs.impala import IMPALA
    from ..algs.parameters import create_ppo_kwargs
    from ..actors.cnn_actors import create_impala_cnn_actor
    from raic2020.local_state_cnn import make_layers, make_layers_impala
    from ..actors.norm_factory import BatchNormFactory

    train_frames = 100e6
    num_envs = 4
    horizon = 64
    actors_per_env = 50
    env_factory = partial(make_env, num_players=4, num_auto_bots=0, num_envs=num_envs)

    alg_class = IMPALA
    alg_params = create_ppo_kwargs(
        train_interval_frames=32 * 1024,
        train_horizon=horizon,
        batch_size=512,
        value_loss_scale=1.0,
        pg_loss_scale=1.0,
        cuda_eval=True,
        cuda_train=True,
        reward_discount=0.99,

        replay_buf_size=512 * 1024,
        replay_end_sampling_factor=0.1,
        grad_clip_norm=None,
        use_pop_art=False,
        reward_scale=0.2,
        kl_pull=0.5,
        eval_model_blend=1.0,
        kl_limit=0.3,
        replay_ratio=3,
        upgo_scale=0.0,
        entropy_loss_scale=0.001,
        activation_norm_scale=0.0,
        reward_reweight_interval=40,
        advantage_discount=0.95,

        ppo_iters=1,
        ppo_policy_clip=5.0,
        ppo_value_clip=5.0,

        model_factory=partial(create_impala_cnn_actor, cnn_kind=partial(make_layers_impala, c_mult=2), add_positional_features=True),
        optimizer_factory=partial(optim.Adam, lr=2e-4),

        # disable_training=True,
        model_init_path='tensorboard\IMPALA_CodeCraft_2020-12-11_08-24-36_[]_st_7e9u8\model_0.pth',
    )
    # trainer_params = dict(
    #     rl_alg_factory=partial(alg_class, **alg_params),
    #     env_factory=env_factory,
    #     alg_name=alg_class.__name__,
    #     tag='[]',
    #     log_root_path=log_path,
    #     log_interval=10000,
    # )
    # run_training(VariableEnvTrainer, trainer_params, alg_params, train_frames)

    trainer_params = dict(
        rl_alg_factory=partial(alg_class, **alg_params),
        env_factory=env_factory,
        alg_name=alg_class.__name__,
        tag='[]',
        log_root_path=log_path,
        log_interval=10000,
        num_archive_models=50,
        archive_save_interval=100_000,
        archive_switch_interval=250 * num_envs * actors_per_env,
        selfplay_prob=1.0,
    )
    #
    # # import cProfile, pstats, io
    # # from pstats import SortKey
    #
    # # pr = cProfile.Profile()
    # # pr.enable()
    run_training(VariableSelfPlayTrainer, trainer_params, alg_params, train_frames)
    # # pr.disable()
    # # s = io.StringIO()
    # # sortby = SortKey.CUMULATIVE
    # # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # # ps.print_stats()
    # # print(s.getvalue())
