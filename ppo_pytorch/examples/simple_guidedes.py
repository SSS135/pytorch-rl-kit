if __name__ == '__main__':
    from .init_vars import *
    from rl_exp.es.guided_es import GuidedES
    from rl_exp.unity_env import UnityVecEnv
    from torch.optim.adamw import AdamW

    exe_path = '/home/alexander/DungeonAI/SimpleArenaDiscreteLinux/DungeonAI.x86_64' if is_linux else \
        'c:\\Users\\Alexander\\projects\\DungeonAI\\Build\\SimpleArenaContinuousR\\DungeonAI'
    env_factory = partial(UnityVecEnv, exe_path, parallel='process')

    alg_class = GuidedES
    alg_params = rl.algs.create_ppo_kwargs(
        5e5,

        num_actors=1,
        horizon=64,
        batch_size=64,
        model_factory=rl.actors.create_ppo_fc_actor,
        cuda_eval=False,
        cuda_train=False,

        grad_buffer_len=32,
        steps_per_update=1,
        es_lr=0.005,
        es_std=0.01,
        es_blend=0.0,

        use_pop_art=False,
        # reward_scale=0.03,
        entropy_loss_scale=1e-3,
        value_loss_scale=0.5,
        reward_discount=0.99,
        replay_buf_size=256 * 1024,
        replay_end_sampling_factor=0.1,
        grad_clip_norm=None,
        kl_pull=0.1,
        vtrace_max_ratio=1.0,
        vtrace_kl_limit=0.2,
        loss_type='impala',
        eval_model_blend=0.1,
        kl_limit=0.2,
        replay_ratio=7,
        upgo_scale=0.0,

        optimizer_factory=partial(AdamW, lr=5e-4, eps=1e-5, weight_decay=1e-5),
    )
    hparams = dict(
    )
    wrap_params = dict(
        tag='[]',
        log_root_path=log_path,
        log_interval=20000,
    )

    rl_alg_test(hparams, wrap_params, alg_class, alg_params, env_factory, num_processes=1, iters=1, frames=5e5)
