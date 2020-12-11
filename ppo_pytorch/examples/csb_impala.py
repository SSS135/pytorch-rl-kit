if __name__ == '__main__':
    from .init_vars import *
    from coders_strike_back import make_gold_env
    from ..common.rl_alg_test import run_training
    from ..common.variable_env.variable_self_play_trainer import VariableSelfPlayTrainer
    from ..algs.impala import IMPALA
    from ..algs.parameters import create_ppo_kwargs
    from ..actors.fc_actors import create_ppo_fc_actor, create_impala_fc_actor
    from ppo_pytorch.actors.rnn_actors import create_impala_rnn_actor
    from ppo_pytorch.common.silu import SiLU

    train_frames = 1000e6
    num_envs = 32
    horizon = 128
    actors_per_env = 4
    env_factory = partial(make_gold_env, pvp=True, num_pods=4, num_envs=num_envs, frame_stack=4, render_first_env=True)

    alg_class = IMPALA
    alg_params = create_ppo_kwargs(
        train_interval_frames=20 * 1024,
        train_horizon=horizon,
        batch_size=1024,
        value_loss_scale=1.0,
        pg_loss_scale=1.0,
        cuda_eval=False,
        cuda_train=True,
        reward_discount=0.99,

        replay_buf_size=256 * 1024,
        replay_end_sampling_factor=1.0,
        grad_clip_norm=2,
        use_pop_art=False,
        reward_scale=1.0,
        kl_pull=0.3,
        eval_model_blend=1.0,
        kl_limit=0.3,
        replay_ratio=0,
        upgo_scale=0.1,
        entropy_loss_scale=0.003,
        activation_norm_scale=0.0,
        reward_reweight_interval=40,
        advantage_discount=0.95,

        ppo_iters=4,
        ppo_policy_clip=0.2,
        ppo_value_clip=0.2,

        model_factory=partial(create_impala_fc_actor, hidden_sizes=(512, 512, 512, 512), activation=SiLU),
        # model_factory=partial(create_impala_rnn_actor, hidden_size=128, num_layers=2),
        # optimizer_factory=partial(GAdam, lr=5e-4, avg_sq_mode='tensor', betas=(0.9, 0.99)),
        optimizer_factory=partial(optim.Adam, lr=2e-4),

        # disable_training=True,
        # model_init_path='tensorboard\IMPALA_CSBPvP_2020-11-25_20-09-54_[]_lbcyr5e3\model_0.pth',
    )
    trainer_params = dict(
        rl_alg_factory=partial(alg_class, **alg_params),
        env_factory=env_factory,
        alg_name=alg_class.__name__,
        tag='[]',
        log_root_path=log_path,
        log_interval=10000,
        num_archive_models=50,
        archive_save_interval=1_000_000,
        archive_switch_interval=250 * num_envs * actors_per_env,
        selfplay_prob=0.9,
    )

    run_training(VariableSelfPlayTrainer, trainer_params, alg_params, train_frames)
