if __name__ == '__main__':
    from .init_vars import *
    from coders_strike_back import make_gold_env
    from ..common.rl_alg_test import run_training
    from ..common.variable_env.variable_self_play_trainer import VariableSelfPlayTrainer
    from ..algs.impala import IMPALA
    from ..algs.parameters import create_ppo_kwargs
    from ..actors.fc_actors import create_ppo_fc_actor
    from ppo_pytorch.common.silu import SiLU

    train_frames = 20e6
    num_envs = 32
    horizon = 128
    actors_per_env = 4
    env_factory = partial(make_gold_env, pvp=True, num_pods=4, num_envs=num_envs, frame_stack=2, render_first_env=True)

    alg_class = IMPALA
    alg_params = create_ppo_kwargs(
        train_frames,

        num_actors=num_envs,
        train_interval_frames=horizon * num_envs * actors_per_env,
        train_horizon=horizon,
        batch_size=512,
        value_loss_scale=2.0,
        q_loss_scale=2.0,
        cuda_eval=True,
        cuda_train=True,

        reward_discount=0.99,
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
        replay_ratio=3,
        upgo_scale=0.5,
        entropy_loss_scale=0.003,
        barron_alpha_c=(1.5, 1.0),
        memory_burn_in_steps=32,
        activation_norm_scale=0.003,
        reward_reweight_interval=40,

        model_factory=partial(create_ppo_fc_actor, hidden_sizes=(256, 256, 256),
                              activation=SiLU, split_policy_value_network=False, use_imagination=False),
        # model_factory=partial(rl.actors.create_ppo_rnn_actor, hidden_size=256, num_layers=3),
        optimizer_factory=partial(optim.Adam, lr=3e-4),

        # model_init_path='tensorboard\IMPALA_CSBPvP_2020-05-15_12-19-42_[ne16_h128_w-randn_mp]_j15vto_z\model_0.pth',
        # disable_training=True,
    )
    trainer_params = dict(
        rl_alg_factory=partial(alg_class, **alg_params),
        env_factory=env_factory,
        alg_name=alg_class.__name__,
        tag='[r3_sp1]',
        log_root_path=log_path,
        log_interval=10000,
        num_archive_models=10,
        archive_save_interval=30_000,
        archive_switch_interval=250 * num_envs * actors_per_env,
        selfplay_prob=1.0,
        rate_agents=True,
    )

    run_training(VariableSelfPlayTrainer, trainer_params, alg_params, train_frames)
