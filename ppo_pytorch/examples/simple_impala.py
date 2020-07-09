from ..common.cartpole_continuous import CartPoleContinuousEnv

if __name__ == '__main__':
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

    train_frames = 2e6
    num_envs = 8
    actors_per_env = 1
    horizon = 64
    env_factory = partial(make_async_env, num_envs=num_envs, env_name='CartPole-v1', frame_stack=1, frame_skip=1)

    alg_class = IMPALA
    alg_params = create_ppo_kwargs(
        train_interval_frames=4 * 512,
        train_horizon=horizon,
        batch_size=512,
        value_loss_scale=1.0,
        pg_loss_scale=1.0,
        cuda_eval=False,
        cuda_train=True,
        reward_discount=0.99,

        replay_buf_size=256 * 1024,
        replay_end_sampling_factor=0.05,
        grad_clip_norm=None,
        use_pop_art=False,
        reward_scale=1.0,
        kl_pull=1.0,
        eval_model_blend=0.05,
        kl_limit=0.3,
        loss_type='impala',
        replay_ratio=7,
        upgo_scale=0.0,
        entropy_loss_scale=0.01,
        memory_burn_in_steps=32,
        activation_norm_scale=0.0,
        reward_reweight_interval=40,
        random_crop_obs=False,
        advantage_discount=1.0,

        # model_factory=partial(create_impala_fc_actor, hidden_sizes=(128, 128), activation=nn.Tanh),
        model_factory=partial(create_impala_rnn_actor, hidden_size=128, num_layers=2),
        # optimizer_factory=partial(GAdam, lr=5e-4, avg_sq_mode='tensor', betas=(0.9, 0.99)),
        optimizer_factory=partial(Adam, lr=3e-4, eps=1e-6),

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
