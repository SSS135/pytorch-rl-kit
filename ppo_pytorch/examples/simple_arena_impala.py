import ppo_pytorch.actors.swish

if __name__ == '__main__':
    from .init_vars import *
    from torch.optim.adamw import AdamW
    from rl_exp.simple_unity_env import UnityVecEnv

    exe_path = r'c:\Users\Alexander\Projects\DungeonAI\Build\SimpleArenaContinuous\DungeonAI'
    env_factory = partial(UnityVecEnv, exe_path, parallel='process')

    alg_class = rl.algs.IMPALA
    alg_params = rl.algs.create_ppo_kwargs(
        20e6,

        num_actors=8,
        horizon=1024,
        batch_size=1024,
        value_loss_scale=5.0,
        cuda_eval=False,
        cuda_train=True,

        # reward_discount=0.997,
        # reward_scale=1.0,

        replay_buf_size=256 * 1024,
        replay_end_sampling_factor=1.0,
        grad_clip_norm=None,
        use_pop_art=False,
        reward_scale=0.3,
        kl_pull=0.1,
        vtrace_max_ratio=1.0,
        vtrace_kl_limit=0.2,
        kl_limit=0.2,
        loss_type='impala',
        eval_model_blend=0.1,
        replay_ratio=7,
        upgo_scale=0.0,
        entropy_loss_scale=0.005,
        barron_alpha_c=(2.0, 1.0),

        model_factory=partial(rl.actors.create_ppo_fc_actor, hidden_sizes=(256, 256, 256),
                              activation=ppo_pytorch.actors.swish.Swish),
        optimizer_factory=partial(AdamW, lr=3e-4),
    )
    hparams = dict(
    )
    wrap_params = dict(
        tag='[newhparam_1024_large-swish_vls5.0]',
        log_root_path=log_path,
        log_interval=20000,
    )

    rl_alg_test(hparams, wrap_params, alg_class, alg_params, env_factory, num_processes=1, iters=1, frames=20e6)
