import numpy as np
from gym import spaces
from gym.envs.classic_control.cartpole import CartPoleEnv
from gym.envs.registration import register

register(
    id='CartPoleNondeterministic-v0',
    entry_point='ppo_pytorch.common.cartpole_nondeterministic:CartPoleNondeterministicEnv',
    max_episode_steps=200,
    reward_threshold=195.0,
)

register(
    id='CartPoleNondeterministic-v1',
    entry_point='ppo_pytorch.common.cartpole_nondeterministic:CartPoleNondeterministicEnv',
    max_episode_steps=500,
    reward_threshold=475.0,
)


class CartPoleNondeterministicEnv(CartPoleEnv):
    """
    Nondeterministic version of CartPole gym environment
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        high = np.array([1, 1])
        self.observation_space = spaces.Box(-high, high)

    def _reset(self, **kwargs):
        observation = super()._reset(**kwargs)
        return self._observation(observation)

    def _step(self, action):
        observation, reward, done, info = super()._step(action)
        return self.observation(observation), reward, done, info

    def observation(self, observation):
        return self._observation(observation)

    def _observation(self, observation):
        return np.array([observation[0], observation[2]])
