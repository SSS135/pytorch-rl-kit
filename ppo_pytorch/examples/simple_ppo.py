from torch.optim import RAdam
from torch_optimizer import Lamb, Adahessian

from ppo_pytorch.actors.norm_factory import LayerNormFactory, RMSNormFactory
from ppo_pytorch.algs.parameters import create_fc_kwargs
from ppo_pytorch.algs.ppo import PPO
from ppo_pytorch.common.env_factory import SimpleVecEnv
from ppo_pytorch.common.env_trainer import EnvTrainer
from ppo_pytorch.common.lookahead import Lookahead
from ppo_pytorch.common.rl_alg_test import run_training
from ..actors.fc_actors import create_ppo_fc_actor
from ..common.cartpole_continuous import CartPoleContinuousEnv

if __name__ == '__main__':
    from .init_vars import *

    train_frames = 2_000_000
    env_factory = partial(SimpleVecEnv, env_name='Hopper-v4')

    alg_class = PPO
    alg_params = create_fc_kwargs(
        num_actors=32,
        horizon=64,
        batch_size=512,
        cuda_eval=True,
        cuda_train=True,

        slow_action_pull=0.1,
        slow_value_pull=0.1,
        ppo_iters=10,
        policy_clip=0.2,
        value_clip=None,
        entropy_loss_scale=3e-4,
        grad_clip_norm=None,
        slow_model_blend=0.1,
        batch_kl_limit=0.1,
        value_loss_scale=0.5,

        model_factory=partial(create_ppo_fc_actor, norm_factory=RMSNormFactory(),
                              hidden_sizes=(512, 512, 512), split_policy_value_network=False),
        optimizer_factory=lambda p: Lookahead(RAdam(p, lr=0.001), k=5, alpha=0.5)
    )
    trainer_params = dict(
        rl_alg_factory=partial(alg_class, **alg_params),
        env_factory=env_factory,
        alg_name=alg_class.__name__,
        tag='[]',
        log_root_path=log_path,
        log_interval=10000,
    )

    run_training(EnvTrainer, trainer_params, alg_params, train_frames)
