if __name__ == '__main__':
    from .init_vars import *
    from torch.optim.adamw import AdamW
    from rl_exp.unity_env import UnityVecEnv

    env_factory = partial(UnityVecEnv,
                          'c:\\Users\\Alexander\\projects\\DungeonAI\\Build\\SimpleArenaDiscrete\\DungeonAI',
                          parallel='process')

    alg_class = rl.algs.IMPALA
    alg_params = rl.algs.create_ppo_kwargs(
        20e6,

        num_actors=8,
        horizon=128,
        batch_size=512,
        value_loss_scale=0.5,
        cuda_eval=True,
        cuda_train=True,

        replay_buf_size=512 * 1024,
        replay_end_sampling_factor=0.01,
        grad_clip_norm=None,
        use_pop_art=True,
        eps_nu_alpha=(0.1, 0.02),
        init_nu_alpha=(1.0, 0.1),
        vtrace_max_ratio=1.0,
        vtrace_kl_limit=0.5,
        loss_type='impala',
        smooth_model_blend=True,
        eval_model_update_interval=100,
        eval_model_blend=0.1,
        kl_limit=0.01,
        replay_ratio=7,
        upgo_scale=0.2,
        entropy_loss_scale=0.0,
        model_factory=partial(rl.actors.create_ppo_fc_actor, hidden_sizes=(256, 256, 256),
                              activation=nn.ReLU),
        optimizer_factory=partial(AdamW, lr=5e-4, eps=1e-5, weight_decay=1e-5),
    )
    hparams = dict(
    )
    wrap_params = dict(
        tag='[nobn_kl0.1_ent0_blend0.1_an_nonvis]',
        log_root_path=log_path,
        log_interval=20000,
    )

    rl_alg_test(hparams, wrap_params, alg_class, alg_params, env_factory, num_processes=1, iters=1, frames=20e6)
