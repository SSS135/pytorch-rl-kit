from ..common.cartpole_continuous import CartPoleContinuousEnv

if __name__ == '__main__':
    # import os
    # os.system('/create_huge_pages.sh')
    #
    # import ray
    # ray.init(include_webui=False, object_store_memory=2 * 1024 * 1024 * 1024 - 1, huge_pages=True, plasma_directory="/mnt/hugepages")

    from .init_vars import *
    from ..algs.parameters import create_ppo_kwargs
    from ppo_pytorch.actors.fc_actors import create_impala_fc_actor
    from ppo_pytorch.common.rl_alg_test import run_training
    from ppo_pytorch.common.variable_env.variable_env_trainer import VariableEnvTrainer
    from ppo_pytorch.common.silu import SiLU
    from ppo_pytorch.algs.impala import IMPALA
    from ppo_pytorch.common.variable_env.gym_to_variable_env import make_async_env
    from optfn.gadam import GAdam
    from ppo_pytorch.actors.rnn_actors import create_impala_rnn_actor
    from torch.optim import Adam

    train_frames = 5e5
    num_envs = 8
    actors_per_env = 1
    horizon = 128
    burnin = 32
    env_factory = partial(make_async_env, num_envs=num_envs, env_name='CartPole-v1', frame_stack=1, frame_skip=1)

    alg_class = IMPALA
    alg_params = create_ppo_kwargs(
        train_interval_frames=8 * 1024,
        train_horizon=horizon,
        batch_size=256,
        value_loss_scale=1.0,
        pg_loss_scale=1.0,
        cuda_eval=False,
        cuda_train=True,
        reward_discount=0.99,

        replay_buf_size=256 * 1024,
        replay_end_sampling_factor=1.0,
        grad_clip_norm=2,
        use_pop_art=False,
        reward_scale=0.1,
        kl_pull=0.3,
        eval_model_blend=1.0,
        kl_limit=0.3,
        replay_ratio=0,
        upgo_scale=0.0,
        entropy_loss_scale=0.005,
        memory_burn_in_steps=burnin,
        activation_norm_scale=0.0,
        reward_reweight_interval=40,
        advantage_discount=0.95,

        ppo_iters=6,
        ppo_policy_clip=0.5,
        ppo_value_clip=0.5,

        # model_factory=partial(create_impala_fc_actor, hidden_sizes=(128, 128), activation=nn.Tanh),
        # model_factory=partial(create_impala_rnn_actor, hidden_size=128, num_layers=2),
        # optimizer_factory=partial(GAdam, lr=5e-4, avg_sq_mode='tensor', betas=(0.9, 0.99)),
        optimizer_factory=partial(Adam, lr=3e-4),

        # model_init_path='tensorboard\IMPALA_CSBPvP_2020-05-15_12-19-42_[ne16_h128_w-randn_mp]_j15vto_z\model_0.pth',
        # disable_training=True,
    )
    trainer_params = dict(
        rl_alg_factory=partial(alg_class, **alg_params),
        env_factory=env_factory,
        alg_name=alg_class.__name__,
        tag='[]',
        log_root_path=log_path,
        log_interval=10000,
    )

    run_training(VariableEnvTrainer, trainer_params, alg_params, train_frames)
