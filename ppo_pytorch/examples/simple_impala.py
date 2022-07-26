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
    import torch_optimizer as optim

    train_frames = 2_000_000
    num_envs = 32
    actors_per_env = 1
    horizon = 64
    burnin = 16
    env_factory = partial(make_async_env, num_envs=num_envs, env_name='Walker2d-v4', frame_stack=1, frame_skip=1)

    alg_class = IMPALA
    alg_params = create_ppo_kwargs(
        train_interval_frames=8 * 1024,
        train_horizon=horizon,
        batch_size=8 * 1024,
        value_loss_scale=1.0,
        pg_loss_scale=1.0,
        cuda_eval=False,
        cuda_train=True,
        reward_discount=0.99,

        replay_buf_size=256 * 1024,
        replay_end_sampling_factor=0.05,
        grad_clip_norm=None,
        use_pop_art=False,
        reward_scale=0.03,
        kl_pull=0.1,
        eval_model_blend=0.1,
        kl_limit=0.3,
        replay_ratio=31,
        upgo_scale=0.0,
        entropy_loss_scale=0.0,
        memory_burn_in_steps=burnin,
        activation_norm_scale=0.0,
        advantage_discount=0.95,

        ppo_iters=1,
        ppo_policy_clip=None,
        ppo_value_clip=None,

        model_factory=partial(create_impala_fc_actor, hidden_sizes=(128, 128), activation=nn.Tanh, noisy_net=False),
        # model_factory=partial(create_impala_rnn_actor, hidden_size=128, num_layers=2),
        # optimizer_factory=partial(GAdam, lr=5e-4, avg_sq_mode='tensor', betas=(0.9, 0.99)),
        # optimizer_factory=partial(Adam, lr=3e-4, eps=1e-6),
        # optimizer_factory=partial(
        #     optim.Adahessian,
        #     lr=0.1,
        #     betas=(0.9, 0.999),
        #     eps=1e-4,
        #     weight_decay=0.0,
        #     hessian_power=0.5,
        # ),
        optimizer_factory=partial(
            optim.Lamb,
            lr=0.005,
        ),
    )
    trainer_params = dict(
        rl_alg_factory=partial(alg_class, **alg_params),
        env_factory=env_factory,
        alg_name=alg_class.__name__,
        tag='[r31]',
        log_root_path=log_path,
        log_interval=10000,
    )

    run_training(VariableEnvTrainer, trainer_params, alg_params, train_frames)
