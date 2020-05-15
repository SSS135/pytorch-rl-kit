if __name__ == '__main__':
    from .init_vars import *
    from coders_strike_back import make_gold_env

    num_envs = 32
    actors_per_env = 1
    env_factory = partial(make_gold_env, pvp=True, num_pods=4, num_envs=num_envs, frame_stack=4, render_first_env=True)

    alg_class = rl.algs.IMPALA
    alg_params = rl.algs.create_ppo_kwargs(
        20e6,

        num_actors=num_envs,
        train_interval_frames=128 * num_envs,
        train_horizon=128,
        batch_size=512,
        value_loss_scale=2.0,
        q_loss_scale=2.0,
        loss_dpg_scale=0.0,
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
        replay_ratio=7,
        upgo_scale=0.5,
        entropy_loss_scale=0.003,
        barron_alpha_c=(1.5, 1.0),
        memory_burn_in_steps=32,
        activation_norm_scale=0.003,
        reward_reweight_interval=40,

        model_factory=partial(rl.actors.create_ppo_fc_actor, hidden_sizes=(256, 256, 256),
                              activation=rl.actors.SiLU, split_policy_value_network=False),
        # model_factory=partial(rl.actors.create_ppo_rnn_actor, hidden_size=256, num_layers=3),
        optimizer_factory=partial(optim.Adam, lr=3e-4),

        # model_init_path='tensorboard\IMPALA_CSBSilverVsScript_2020-05-14_02-22-16_[rewsm1_rewsep]_dx_h4owk\model_0.pth',
        # disable_training=True,
    )
    hparams = dict(
    )
    wrap_params = dict(
        tag='[mp]',
        log_root_path=log_path,
        log_interval=10000,
    )

    rl_alg_test(hparams, wrap_params, alg_class, alg_params, env_factory, variable_env=True, frames=20e6)
