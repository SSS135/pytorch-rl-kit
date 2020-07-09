if __name__ == '__main__':
    from .init_vars import *
    from ppo_pytorch.actors.cnn_actors import create_ppo_cnn_actor
    from ppo_pytorch.algs.parameters import create_atari_kwargs
    from ppo_pytorch.algs.ppo import PPO
    from ppo_pytorch.common.env_factory import AtariVecEnv

    env_factory = partial(AtariVecEnv, 'BreakoutNoFrameskip-v4', parallel='process',
                          episode_life=True, clip_rewards=True)

    alg_class = PPO
    alg_params = create_atari_kwargs(
        num_actors=8,
        horizon=256,
        batch_size=256,
        value_loss_scale=0.5,
        cuda_eval=True,
        cuda_train=True,

        use_pop_art=False,
        reward_scale=1.0,
        ppo_iters=3,
        constraint='clip',
        spu_dis_agg_lam=(0.01, 0.01 / 1.3, 2.0),
        value_clip=0.2,
        policy_clip=0.2,
        entropy_loss_scale=0.005,
        grad_clip_norm=None,
        squash_values=False,

        optimizer_factory=partial(optim.Adam, lr=3e-4, eps=1e-6),
        model_factory=partial(create_ppo_cnn_actor, cnn_kind='normal'),
    )
    hparams = dict(
    )
    wrap_params = dict(
        tag='[clip_vclip_iter3]',
        log_root_path=log_path,
        log_interval=20000,
    )

    rl_alg_test(hparams, wrap_params, alg_class, alg_params, env_factory, num_processes=1, iters=1, frames=50e6)
