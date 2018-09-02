from concurrent.futures import ThreadPoolExecutor

import numpy as np
from baselines.common.vec_env import VecEnv
import os


class ThreadingVecEnv(VecEnv):
    def __init__(self, env_fns):
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count())
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.ts = np.zeros(len(self.envs), dtype='int')

    def step(self, action_n):
        results = self.executor.map(lambda t: t[1].step(t[0]), zip(action_n, self.envs))
        obs, rews, dones, infos = map(np.array, zip(*results))
        self.ts += 1
        for (i, done) in enumerate(dones):
            if done:
                obs[i] = self.envs[i].reset()
                self.ts[i] = 0
        return np.array(obs), np.array(rews), np.array(dones), infos

    def reset(self):
        results = [env.reset() for env in self.envs]
        return np.array(results)

    @property
    def num_envs(self):
        return len(self.envs)