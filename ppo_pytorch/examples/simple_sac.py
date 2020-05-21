from ppo_pytorch.actors.fc_actors import create_sac_fc_actor
from ppo_pytorch.actors.silu import SiLU

if __name__ == '__main__':
    from .init_vars import *
    from ..algs.parameters import create_sac_fc_kwargs
    from ppo_pytorch.algs.sac import SAC
    from ppo_pytorch.common.env_factory import SimpleVecEnv
    import ppo_pytorch.common.cartpole_continuous
    import pybullet_envs

    train_frames = 2e6
    env_factory = partial(SimpleVecEnv, 'AntBulletEnv-v0', parallel='dummy')

    alg_class = SAC
    alg_params = create_sac_fc_kwargs(
        train_frames,
        cuda_eval=False,
        cuda_train=True,
        num_actors=8,
        train_interval=16,
        batch_size=512,
        num_batches=16,
        kl_pull=0.1,
        replay_buffer_size=128*1024,
        replay_end_sampling_factor=0.05,
        reward_scale=0.1,
        rollout_length=32,
        vtrace_kl_limit=0.5,
        actor_update_interval=2,
        entropy_scale=0.1,
        actor_optimizer_factory=partial(optim.Adam, lr=5e-4),
        critic_optimizer_factory=partial(optim.Adam, lr=5e-4),
        model_factory=partial(create_sac_fc_actor, activation=SiLU),
    )
    hparams = dict(
    )
    wrap_params = dict(
        tag='[silu_lr5_randac0.05_kllim0.5_klpull0.1_bs512_resf0.05_ui8]',
        log_root_path=log_path,
        log_interval=10000,
    )

    rl_alg_test(hparams, wrap_params, alg_class, alg_params, env_factory, num_processes=1, iters=1, frames=train_frames)
