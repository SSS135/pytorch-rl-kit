if __name__ == '__main__':
    from .init_vars import *
    import pybullet_envs
    from optfn.gadam import GAdam

    env_factory = partial(rl.common.SimpleVecEnv, 'AntBulletEnv-v0', parallel='dummy')

    alg_class = rl.algs.IMPALA
    alg_params = rl.algs.create_ppo_kwargs(
        4e6,

        num_actors=8,
        horizon=64,
        batch_size=128,
        model_factory=rl.actors.create_ppo_fc_actor,
        cuda_eval=False,
        cuda_train=False,

        replay_buf_size=2 * 128 * 1024,
        replay_end_sampling_factor=0.1,
        grad_clip_norm=None,
        use_pop_art=True,
        eps_nu_alpha=(0.1, 0.005),
        init_nu_alpha=(1.0, 5.0),
        vtrace_max_ratio=2.0,
        vtrace_kl_limit=0.5,
        loss_type='v_mpo',
        eval_model_update_interval=10,
        replay_ratio=7,
        upgo_scale=0.2,
        optimizer_factory=partial(optim.Adam, lr=5e-4, eps=1e-5),
    )
    hparams = dict(
    )
    wrap_params = dict(
        tag='[blend_0.2]',
        log_root_path=log_path,
        log_interval=10000,
    )

    rl_alg_test(hparams, wrap_params, alg_class, alg_params, env_factory, num_processes=1, iters=1, frames=4e6)
