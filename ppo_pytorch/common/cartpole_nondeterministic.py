import numpy as np
from gymnasium import spaces
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from gymnasium.envs.registration import register

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

    def reset(self, **kwargs):
        observation = super().reset(**kwargs)
        return self.observation(observation)

    def step(self, action):
        observation, reward, done, info = super().step(action)
        return self.observation(observation), reward, done, info

    def observation(self, observation):
        return np.array([observation[0], observation[2]])
