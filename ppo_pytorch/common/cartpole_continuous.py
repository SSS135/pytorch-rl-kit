import numpy as np
from gym import spaces
from gym.envs.classic_control.cartpole import CartPoleEnv
from gym.envs.registration import register

register(
    id='CartPoleContinuous-v0',
    entry_point='ppo_pytorch.common.cartpole_continuous:CartPoleContinuousEnv',
    max_episode_steps=200,
    reward_threshold=195.0,
)

register(
    id='CartPoleContinuous-v1',
    entry_point='ppo_pytorch.common.cartpole_continuous:CartPoleContinuousEnv',
    max_episode_steps=500,
    reward_threshold=475.0,
)

register(
    id='CartPoleContinuous-v2',
    entry_point='ppo_pytorch.common.cartpole_continuous:CartPoleContinuousEnv',
    max_episode_steps=10000,
    reward_threshold=9500.0,
)

register(
    id='CartPole-v2',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    max_episode_steps=10000,
    reward_threshold=9500.0,
)


class CartPoleContinuousEnv(CartPoleEnv):
    """
    Continuous version of CartPole gym environment
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._set_action_space(False)

    def step(self, action):
        self._set_action_space(True)
        action = np.clip(np.asscalar(action), -1, 1)
        force_bak = self.force_mag
        self.force_mag = force_bak * action
        res = super().step(1)
        self.force_mag = force_bak
        self._set_action_space(False)
        return res

    def _set_action_space(self, discrete):
        if discrete:
            self.action_space = spaces.Discrete(2)
        else:
            high = np.array([1])
            self.action_space = spaces.Box(-high, high, dtype=np.float32)
