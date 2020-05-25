import pybullet_envs

if __name__ == '__main__':
    from .init_vars import *
    from optfn.gadam import GAdam
    from torch.optim.adamw import AdamW
    from ..algs.parameters import create_ppo_kwargs
    from ppo_pytorch.actors.fc_actors import create_ppo_fc_actor
    from ppo_pytorch.common.rl_alg_test import run_training
    from ppo_pytorch.common.variable_env.variable_env_trainer import VariableEnvTrainer
    from ppo_pytorch.actors.silu import SiLU
    from ppo_pytorch.algs.impala import IMPALA
    import ppo_pytorch.common.cartpole_continuous
    from ppo_pytorch.common.variable_env.gym_to_variable_env import make_simple_env

    train_frames = 2e6
    num_envs = 16
    actors_per_env = 1
    horizon = 128
    env_factory = partial(make_simple_env, num_envs=num_envs, env_name='AntBulletEnv-v0', frame_stack=1, frame_skip=2)

    alg_class = IMPALA
    alg_params = create_ppo_kwargs(
        train_frames,

        num_actors=num_envs,
        train_interval_frames=horizon * num_envs * actors_per_env,
        train_horizon=horizon,
        batch_size=512,
        value_loss_scale=2.0,
        q_loss_scale=2.0,
        cuda_eval=True,
        cuda_train=True,

        reward_discount=0.99,
        # reward_scale=1.0,

        replay_buf_size=512 * 1024,
        replay_end_sampling_factor=0.05,
        grad_clip_norm=None,
        use_pop_art=False,
        reward_scale=0.05,
        kl_pull=0.05,
        eval_model_blend=0.05,
        vtrace_max_ratio=1.0,
        vtrace_kl_limit=1.0,
        kl_limit=0.2,
        loss_type='impala',
        replay_ratio=3,
        upgo_scale=0.5,
        entropy_loss_scale=0.005,
        barron_alpha_c=(1.5, 1.0),
        memory_burn_in_steps=32,
        activation_norm_scale=0.0,
        reward_reweight_interval=40,

        model_factory=partial(create_ppo_fc_actor, hidden_sizes=(256, 256, 256), activation=SiLU,
                              split_policy_value_network=False, use_imagination=False),
        # model_factory=partial(rl.actors.create_ppo_rnn_actor, hidden_size=256, num_layers=3),
        optimizer_factory=partial(optim.Adam, lr=1e-3),

        # model_init_path='tensorboard\IMPALA_CSBPvP_2020-05-15_12-19-42_[ne16_h128_w-randn_mp]_j15vto_z\model_0.pth',
        # disable_training=True,
    )
    trainer_params = dict(
        rl_alg_factory=partial(alg_class, **alg_params),
        env_factory=env_factory,
        alg_name=alg_class.__name__,
        tag='[ent005_lr10_an0_fs2]',
        log_root_path=log_path,
        log_interval=10000,
    )

    run_training(VariableEnvTrainer, trainer_params, alg_params, train_frames)
