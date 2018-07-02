"""
Environments and wrappers for Sonic training.
https://github.com/openai/retro-baselines/blob/master/agents/sonic_util.py
"""
import gzip
import os
import random

import gym
import numpy as np
import retro
from retro.retro_env import RetroEnv


# from baselines.common.atari_wrappers import WarpFrame, FrameStack
# import gym_remote.client as grc
#
# def make_env(stack=True, scale_rew=True):
#     """
#     Create an environment with some standard wrappers.
#     """
#     env = grc.RemoteEnv('tmp/sock')
#     env = SonicDiscretizer(env)
#     if scale_rew:
#         env = RewardScaler(env)
#     env = WarpFrame(env)
#     if stack:
#         env = FrameStack(env, 4)
#     return env


class SonicDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env):
        super(SonicDiscretizer, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [[], ['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
                   ['DOWN', 'B'], ['B']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a): # pylint: disable=W0221
        return self._actions[a].copy()


class AllowBacktracking(gym.Wrapper):
    """
    Use deltas in max(X) as the reward, rather than deltas
    in X. This way, agents are not discouraged too heavily
    from exploring backwards if there is no way to advance
    head-on in the level.
    """
    def __init__(self, env):
        super(AllowBacktracking, self).__init__(env)
        self._cur_x = 0
        self._max_x = 0

    def reset(self, **kwargs): # pylint: disable=E0202
        self._cur_x = 0
        self._max_x = 0
        return self.env.reset(**kwargs)

    def step(self, action): # pylint: disable=E0202
        obs, rew, done, info = self.env.step(action)
        self._cur_x += rew
        rew = max(0, self._cur_x - self._max_x)
        self._max_x = max(self._max_x, self._cur_x)
        return obs, rew, done, info


class ChangeStateAtRestart(gym.Wrapper):
    def __init__(self, env, state_names):
        self.state_names = state_names
        super().__init__(env)

    def reset(self, **kwargs):
        env: RetroEnv = self.unwrapped
        env.statename = state = random.choice(self.state_names)
        game_path = retro.get_game_path(env.gamename)
        if not state.endswith('.state'):
            state += '.state'
        with gzip.open(os.path.join(game_path, state), 'rb') as fh:
            env.initial_state = fh.read()
        return self.env.reset(**kwargs)


sonic_1_train_levels = [
    'GreenHillZone.Act1',
    # 'GreenHillZone.Act2',
    'GreenHillZone.Act3',
    'LabyrinthZone.Act1',
    'LabyrinthZone.Act2',
    'LabyrinthZone.Act3',
    'MarbleZone.Act1',
    'MarbleZone.Act2',
    'MarbleZone.Act3',
    # 'ScrapBrainZone.Act1',
    'ScrapBrainZone.Act2',
    # 'SpringYardZone.Act1',
    'SpringYardZone.Act2',
    'SpringYardZone.Act3',
    'StarLightZone.Act1',
    'StarLightZone.Act2',
    # 'StarLightZone.Act3',
]


sonic_1_test_levels = [
    'SpringYardZone.Act1',
    'GreenHillZone.Act2',
    'StarLightZone.Act3',
    'ScrapBrainZone.Act1',
]

sonic_2_train_levels = [
    'AquaticRuinZone.Act1',
    'AquaticRuinZone.Act2',
    'CasinoNightZone.Act1',
    # 'CasinoNightZone.Act2',
    'ChemicalPlantZone.Act1',
    'ChemicalPlantZone.Act2',
    'EmeraldHillZone.Act1',
    'EmeraldHillZone.Act2',
    'HillTopZone.Act1',
    # 'HillTopZone.Act2',
    'MetropolisZone.Act1',
    'MetropolisZone.Act2',
    # 'MetropolisZone.Act3',
    'MysticCaveZone.Act1',
    'MysticCaveZone.Act2',
    'OilOceanZone.Act1',
    'OilOceanZone.Act2',
    'WingFortressZone',
]


sonic_2_test_levels = [
    'MetropolisZone.Act3',
    'CasinoNightZone.Act2',
    'HillTopZone.Act2',
]

sonic_3_train_levels = [
    'AngelIslandZone.Act1',
    # 'AngelIslandZone.Act2',
    'CarnivalNightZone.Act1',
    'CarnivalNightZone.Act2',
    'DeathEggZone.Act1',
    'DeathEggZone.Act2',
    'FlyingBatteryZone.Act1',
    # 'FlyingBatteryZone.Act2',
    'HiddenPalaceZone',
    # 'HydrocityZone.Act1',
    'HydrocityZone.Act2',
    'IcecapZone.Act1',
    'IcecapZone.Act2',
    'LaunchBaseZone.Act1',
    'LaunchBaseZone.Act2',
    # 'LavaReefZone.Act1',
    'LavaReefZone.Act2',
    'MarbleGardenZone.Act1',
    'MarbleGardenZone.Act2',
    'MushroomHillZone.Act1',
    'MushroomHillZone.Act2',
    'SandopolisZone.Act1',
    'SandopolisZone.Act2',
]


sonic_3_test_levels = [
    'LavaReefZone.Act1',
    'FlyingBatteryZone.Act2',
    'HydrocityZone.Act1',
    'AngelIslandZone.Act2',
]