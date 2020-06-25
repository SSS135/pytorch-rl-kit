from optfn.gadam import GAdam
from torch.optim import Adam

if __name__ == '__main__':
    from .init_vars import *
    from ..common.rl_alg_test import run_training
    from ..algs.impala import IMPALA
    from ..algs.parameters import create_ppo_kwargs
    from ppo_pytorch.actors.cnn_actors import create_impala_cnn_actor
    from ppo_pytorch.common.variable_env.variable_env_trainer import VariableEnvTrainer
    from ppo_pytorch.common.variable_env.atari_async_env import make_atari_async_env

    train_frames = 10e6
    num_envs = 16
    horizon = 64

    env_factory = partial(make_atari_async_env, num_envs=num_envs, env_name='BreakoutNoFrameskip-v4',
                          episode_life=False, clip_rewards=False)

    alg_class = IMPALA
    alg_params = create_ppo_kwargs(
        train_interval_frames=4 * 512,
        train_horizon=horizon,
        batch_size=512,
        value_loss_scale=1.0,
        pg_loss_scale=1.0,
        cuda_eval=True,
        cuda_train=True,
        reward_discount=0.99,

        replay_buf_size=256 * 1024,
        replay_end_sampling_factor=0.05,
        grad_clip_norm=None,
        use_pop_art=False,
        reward_scale=1.0,
        kl_pull=1.0,
        eval_model_blend=0.02,
        vtrace_kl_limit=0.5,
        kl_limit=0.5,
        loss_type='impala',
        replay_ratio=3,
        upgo_scale=0.0,
        entropy_loss_scale=0.01,
        memory_burn_in_steps=32,
        activation_norm_scale=0.0,
        reward_reweight_interval=40,
        random_crop_obs=False,
        advantage_discount=1.0,
        target_vmpo_temp=0.1,

        # optimizer_factory=partial(GAdam, lr=5e-4, avg_sq_mode='tensor', betas=(0.9, 0.99)),
        optimizer_factory=partial(Adam, lr=3e-4, eps=1e-6),
        model_factory=partial(create_impala_cnn_actor, cnn_kind='normal'),
        # model_factory=partial(create_impala_rnn_actor, hidden_size=256, num_layers=3),
        # model_factory=partial(create_impala_attention_actor, num_units=9, unit_size=7,
        #                       hidden_size=256, activation=SiLU, split_policy_value_network=True),
        # model_factory=partial(create_impala_fc_actor, hidden_sizes=(256, 256, 256),
        #                       activation=SiLU, use_imagination=False),
    )
    trainer_params = dict(
        tag='[ad1_blend0.02_pull1_vls1_r3_kllim0.5_vtkllim0.5]',
        log_root_path=log_path,
        log_interval=20000,
        rl_alg_factory=partial(alg_class, **alg_params),
        env_factory=env_factory,
        alg_name=alg_class.__name__,
    )

    run_training(VariableEnvTrainer, trainer_params, alg_params, train_frames)
