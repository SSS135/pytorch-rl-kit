if __name__ == '__main__':
    from .init_vars import *
    from ppo_pytorch.actors.cnn_actors import create_ppo_cnn_actor
    from ppo_pytorch.algs.parameters import create_atari_kwargs
    from ppo_pytorch.algs.ppo import PPO
    from ppo_pytorch.common.env_factory import AtariVecEnv
    from optfn.gadam import GAdam
    from ppo_pytorch.common.env_trainer import EnvTrainer
    from ppo_pytorch.common.rl_alg_test import run_training

    env_factory = partial(AtariVecEnv, 'BreakoutNoFrameskip-v4', parallel='process',
                          episode_life=True, clip_rewards=True)

    alg_class = PPO
    alg_params = create_atari_kwargs(
        num_actors=16,
        horizon=256,
        batch_size=512,
        value_loss_scale=0.5,
        cuda_eval=True,
        cuda_train=True,

        use_pop_art=False,
        reward_scale=1.0,
        ppo_iters=3,
        value_clip=0.2,
        policy_clip=0.2,
        kl_pull=0.3,
        batch_kl_limit=1.0,
        entropy_loss_scale=0.005,
        grad_clip_norm=20,
        squash_values=False,

        optimizer_factory=partial(optim.Adam, lr=3e-4, eps=1e-6),
        # optimizer_factory=partial(GAdam, lr=3e-4, avg_sq_mode='tensor'),
        model_factory=partial(create_ppo_cnn_actor, cnn_kind='normal'),
    )
    trainer_params = dict(
        tag='[ac16_bs512_msekl_gcn20_kl0.3_kllim1_clip_vclip_iter3]',
        log_root_path=log_path,
        log_interval=20000,
        rl_alg_factory=partial(alg_class, **alg_params),
        env_factory=env_factory,
        alg_name=alg_class.__name__,
    )

    run_training(EnvTrainer, trainer_params, alg_params, 50e6)
