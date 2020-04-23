if __name__ == '__main__':
    from .init_vars import *
    from torch.optim.adamw import AdamW
    from rl_exp.es.snes_alg import SNES

    env_factory = partial(rl.common.SimpleVecEnv, 'CartPole-v1', parallel='dummy')

    alg_class = SNES
    alg_params = rl.algs.create_ppo_kwargs(
        None,

        num_actors=1,
        lr=0.1,
        initial_noise_scale=0.03,
        std_step=0.0,
        pop_size=8,
    )
    hparams = dict(
    )
    wrap_params = dict(
        tag='[]',
        log_root_path=log_path,
        log_interval=10000,
    )

    rl_alg_test(hparams, wrap_params, alg_class, alg_params, env_factory, num_processes=1, iters=1, frames=2e6)
