import numpy as np
import ray

from ppo_pytorch.common.variable_env.async_ray_env import RemoteEnv


class RayVecEnv:
    def __init__(self, env_fns):
        self.envs = [RemoteEnv.remote(ef, i) for i, ef in enumerate(env_fns)]
        self.observation_space, self.action_space = ray.get(self.envs[0].get_spaces.remote())

    def step(self, actions):
        results = ray.get([env.step.remote(a) for env, a in zip(self.envs, actions)])
        obs, rews, dones, infos = map(np.array, zip(*results))
        for (i, done) in enumerate(dones):
            if done:
                obs[i] = ray.get(self.envs[i].reset.remote())
        return np.array(obs), np.array(rews), np.array(dones), infos

    def reset(self):
        results = ray.get([env.reset.remote() for env in self.envs])
        return np.array(results)

    @property
    def num_envs(self):
        return len(self.envs)