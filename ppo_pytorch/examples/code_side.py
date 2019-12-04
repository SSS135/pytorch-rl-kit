import raic2019

if __name__ == '__main__':
    from .init_vars import *

    env_factory = partial(rl.common.SimpleVecEnv, 'CodeSideAI10-v0', parallel='process')

    alg_class = rl.algs.IMPALA
    alg_params = rl.algs.create_fc_kwargs(
        2e6,
        grad_clip_norm=None,
        #     use_pop_art=True,
        eps_nu_alpha=(1.7639, 0.02),
        init_nu_alpha=(1.0, 5.0),
        vtrace_max_ratio=1.0,
        vtrace_kl_limit=0.1,
        loss_type='impala',
        eval_model_update_interval=5,
        replay_ratio=24,
        reward_discount=0.999,
        num_actors=16,
        model_factory=raic2019.create_actor,
        cuda_eval=False,
        cuda_train=True,
        optimizer_factory=partial(optim.Adam, lr=5e-4, betas=(0.9, 0.99), eps=1e-5),
        # model_init_path='./tensorboard/IMPALA_CodeSideAI-v0__2019-12-01_14-30-35_orq7nn82/model_0.pth',
        # replay_buf_size=240 * 1024,
        # horizon=256,
        # batch_size=256,
        # num_actors=8,
        # upgo_scale=0.0,
        # reward_scale=0.1,
    )
    hparams = dict(
    )
    wrap_params = dict(
        log_root_path=log_path,
    )

    rl_alg_test(hparams, wrap_params, alg_class, alg_params, env_factory, num_processes=1, iters=1, frames=2e6)
