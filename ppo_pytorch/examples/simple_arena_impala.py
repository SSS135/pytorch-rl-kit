if __name__ == '__main__':
    from .init_vars import *
    from torch.optim.adamw import AdamW
    from rl_exp.unity_env import UnityVecEnv

    exe_path = '/home/alexander/DungeonAI/SimpleArenaDiscreteLinux/DungeonAI.x86_64' if is_linux else \
        'c:\\Users\\Alexander\\projects\\DungeonAI\\Build\\SimpleArenaContinuous\\DungeonAI'
    env_factory = partial(UnityVecEnv, exe_path, parallel='process')

    alg_class = rl.algs.IMPALA
    alg_params = rl.algs.create_ppo_kwargs(
        20e6,

        num_actors=16,
        horizon=64,
        batch_size=512,
        value_loss_scale=0.5,
        cuda_eval=True,
        cuda_train=True,

        replay_buf_size=256 * 1024,
        replay_end_sampling_factor=0.1,
        grad_clip_norm=None,
        use_pop_art=True,
        kl_pull=0.1,
        vtrace_max_ratio=1.0,
        vtrace_kl_limit=0.2,
        loss_type='impala',
        eval_model_blend=0.1,
        kl_limit=0.2,
        replay_ratio=7,
        upgo_scale=0.0,
        entropy_loss_scale=1e-3,

        model_factory=partial(rl.actors.create_ppo_fc_actor, hidden_sizes=(256, 256, 256),
                              activation=nn.ReLU),
        optimizer_factory=partial(AdamW, lr=1e-4, eps=1e-5),
    )
    hparams = dict(
    )
    wrap_params = dict(
        tag='[lr1_anw0]',
        log_root_path=log_path,
        log_interval=20000,
    )

    rl_alg_test(hparams, wrap_params, alg_class, alg_params, env_factory, num_processes=1, iters=1, frames=20e6)
