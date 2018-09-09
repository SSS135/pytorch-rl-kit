from functools import partial
from multiprocessing.dummy import Pool

import numpy as np
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

from .atari_wrappers import ScaledFloatFrame, FrameStack
from .env_factory import NamedVecEnv, SimplifyFrame, ChannelTranspose
from .monitor import Monitor
from .sonic_utils import SonicDiscretizer, AllowBacktracking, sonic_1_train_levels, \
    sonic_2_train_levels, sonic_3_train_levels, ChangeStateAtRestart


class SonicVecEnv(NamedVecEnv):
    def __init__(self, game, state, scale=True, frame_stack=False, grayscale=True):
        self.scale = scale
        self.state = state
        self.frame_stack = frame_stack
        self.grayscale = grayscale
        super().__init__(game)

    def get_env_fn(self):
        def make(game, state, scale, frame_stack, grayscale):
            from retro_contest.local import make
            env = make(game, state)
            env = Monitor(env)
            env = SonicDiscretizer(env)
            env = AllowBacktracking(env)
            env = Monitor(env)
            env = SimplifyFrame(env, 84, grayscale)
            env = ChannelTranspose(env)
            if scale:
                env = ScaledFloatFrame(env)
            if frame_stack:
                env = FrameStack(env, 4)
            return env
        return partial(make, self.env_name, self.state, self.scale, self.frame_stack, self.grayscale)


class JointSonicVecEnv:
    sonic_names = ('SonicTheHedgehog-Genesis', 'SonicTheHedgehog2-Genesis', 'SonicAndKnuckles3-Genesis')

    def __init__(self, states='train', scale=True, frame_stack=False, grayscale=True):
        if states == 'train':
            states = [sonic_1_train_levels, sonic_2_train_levels, sonic_3_train_levels]
        assert len(states) == 3
        self.states = states
        self.scale = scale
        self.frame_stack = frame_stack
        self.grayscale = grayscale
        self.env_name = 'Sonic123'
        self.pool = Pool(3)
        self.subproc_envs = None
        self.num_envs = None

        env = self.get_env_fn(self.sonic_names[0], states[0])()
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        env.close()

    def get_env_fn(self, game, states):
        def make(game, states, scale, frame_stack, grayscale):
            from retro_contest.local import make
            env = make(game, states[0])
            env = Monitor(env)
            env = SonicDiscretizer(env)
            env = AllowBacktracking(env)
            env = Monitor(env)
            env = ChangeStateAtRestart(env, states)
            env = SimplifyFrame(env, 84, grayscale)
            env = ChannelTranspose(env)
            if scale:
                env = ScaledFloatFrame(env)
            if frame_stack:
                env = FrameStack(env, 4)
            return env
        return partial(make, game, states, self.scale, self.frame_stack, self.grayscale)

    def set_num_envs(self, num_envs):
        assert num_envs % 3 == 0
        if self.subproc_envs is not None:
            for e in self.subproc_envs:
                e.close()
        self.num_envs = num_envs
        self.subproc_envs = []
        for game, states in zip(self.sonic_names, self.states):
            self.env_name = game
            self.subproc_envs.append(SubprocVecEnv([self.get_env_fn(game, states)] * (num_envs // 3)))
        self.env_name = 'Sonic123'

    def step(self, actions):
        res = self.pool.starmap(lambda env, a: env.step(a), zip(self.subproc_envs, np.split(actions, 3, axis=0)))
        states, rewards, dones, infos = zip(*res)
        return np.concatenate(states), np.concatenate(rewards), np.concatenate(dones), [inf for arr in infos for inf in arr]

    def reset(self):
        return np.concatenate([env.reset() for env in self.subproc_envs], axis=0)