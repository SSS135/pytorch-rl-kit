if __name__ == '__main__':
    from .init_vars import *

    env_factory = partial(rl.common.AtariVecEnv, 'BreakoutNoFrameskip-v4', parallel='process')

    alg_class = rl.algs.PPO
    alg_params = rl.algs.create_atari_kwargs(
        10e6,
        # constraint='spu',
        # ppo_iters=6,
    )
    hparams = dict(
    )
    wrap_params = dict(
        log_root_path=log_path,
        log_interval=20000,
    )

    rl_alg_test(hparams, wrap_params, alg_class, alg_params, env_factory, num_processes=1, iters=1, frames=10e6)
