import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from stable_baselines3.common.vec_env import VecEnv


class ThreadingVecEnv:
    def __init__(self, env_fns):
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count())
        with ThreadPoolExecutor(max_workers=8 * os.cpu_count()) as start_exec:
            self.envs = list(start_exec.map(lambda fn: fn(), env_fns))
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
                obs[i] = self.envs[i].reset()[0]
                self.ts[i] = 0
        return np.array(obs), np.array(rews), np.array(dones), infos

    def reset(self):
        results = [env.reset()[0] for env in self.envs]
        return np.array(results)

    @property
    def num_envs(self):
        return len(self.envs)