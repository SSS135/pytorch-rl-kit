import numpy as np
from gym import spaces
from gym.envs.classic_control.acrobot import AcrobotEnv
from gym.envs.registration import register

register(
    id='AcrobotContinuous-v1',
    entry_point='ppo_pytorch.common.acrobot_continuous:AcrobotContinuousEnv',
    max_episode_steps=500,
)


class AcrobotContinuousEnv(AcrobotEnv):
    """
    Continuous version of Acrobot gym environment
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._set_action_space(False)

    def step(self, action):
        self._set_action_space(True)
        action = np.clip(np.asscalar(action), -1, 1)
        torque_bak = self.AVAIL_TORQUE[0]
        self.AVAIL_TORQUE[0] = action
        res = super().step(0)
        self.AVAIL_TORQUE[0] = torque_bak
        self._set_action_space(False)
        return res

    def _set_action_space(self, discrete):
        if discrete:
            self.action_space = spaces.Discrete(3)
        else:
            high = np.array([1])
            self.action_space = spaces.Box(-high, high, dtype=np.float32)
