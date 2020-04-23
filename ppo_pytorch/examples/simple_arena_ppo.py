if __name__ == '__main__':
    from .init_vars import *
    from torch.optim.adamw import AdamW
    from rl_exp.unity_env import UnityVecEnv

    exe_path = '/home/alexander/DungeonAI/SimpleArenaDiscreteLinux/DungeonAI.x86_64' if is_linux else \
        'c:\\Users\\Alexander\\projects\\DungeonAI\\Build\\SimpleArenaContinuous\\DungeonAI'
    env_factory = partial(UnityVecEnv, exe_path, parallel='process')

    alg_class = rl.algs.PPO
    alg_params = rl.algs.create_ppo_kwargs(
        20e6,

        num_actors=16,
        horizon=64,
        batch_size=512,
        value_loss_scale=0.5,
        cuda_eval=True,
        cuda_train=True,

        use_pop_art=False,
        reward_scale=0.2,
        ppo_iters=9,
        constraint='spu',
        spu_dis_agg_lam=(0.03, 0.02, 1.0),
        value_clip=0.5,

        model_factory=partial(rl.actors.create_ppo_fc_actor, hidden_sizes=(256, 256, 256),
                              activation=nn.ReLU),
        optimizer_factory=partial(AdamW, lr=1e-4, eps=1e-6),
    )
    hparams = dict(
    )
    wrap_params = dict(
        tag='[logitclamp_s3-2]',
        log_root_path=log_path,
        log_interval=20000,
    )

    rl_alg_test(hparams, wrap_params, alg_class, alg_params, env_factory, num_processes=1, iters=1, frames=20e6)
