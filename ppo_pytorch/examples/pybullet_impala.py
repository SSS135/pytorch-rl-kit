if __name__ == '__main__':
    from .init_vars import *
    import pybullet_envs
    from optfn.gadam import GAdam
    from torch.optim import AdamW

    env_factory = partial(rl.common.SimpleVecEnv, 'AntBulletEnv-v0', parallel='dummy')

    alg_class = rl.algs.IMPALA
    alg_params = rl.algs.create_ppo_kwargs(
        4e6,

        num_actors=8,
        horizon=64,
        batch_size=128,
        value_loss_scale=0.5,
        cuda_eval=True,
        cuda_train=True,

        replay_buf_size=512 * 1024,
        replay_end_sampling_factor=0.01,
        grad_clip_norm=None,
        use_pop_art=True,
        eps_nu_alpha=(0.1, 0.02),
        init_nu_alpha=(1.0, 1.0),
        vtrace_max_ratio=2.0,
        vtrace_kl_limit=1.0,
        loss_type='impala',
        smooth_model_blend=True,
        eval_model_update_interval=100,
        eval_model_blend=0.01,
        replay_ratio=7,
        upgo_scale=0.2,
        entropy_loss_scale=1e-3,
        model_factory=rl.actors.create_ppo_fc_actor,
        optimizer_factory=partial(AdamW, lr=5e-4, eps=1e-5),
    )
    hparams = dict(
    )
    wrap_params = dict(
        tag='[alpha_no-kl-mask]',
        log_root_path=log_path,
        log_interval=10000,
    )

    rl_alg_test(hparams, wrap_params, alg_class, alg_params, env_factory, num_processes=1, iters=1, frames=4e6)
