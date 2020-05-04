if __name__ == '__main__':
    from .init_vars import *

    env_factory = partial(rl.common.SimpleVecEnv, 'CartPoleContinuous-v1', parallel='dummy')

    alg_class = rl.algs.SAC
    alg_params = rl.algs.create_sac_fc_kwargs(
        5e5,
        cuda_eval=False,
        cuda_train=True,
        num_actors=8,
        train_interval=64,
        batch_size=128,
        num_batches=8,
        kl_pull=0.0,
        reward_scale=0.1,
    )
    hparams = dict(
    )
    wrap_params = dict(
        tag='[bs32]',
        log_root_path=log_path,
        log_interval=10000,
    )

    rl_alg_test(hparams, wrap_params, alg_class, alg_params, env_factory, num_processes=1, iters=1, frames=5e5)
