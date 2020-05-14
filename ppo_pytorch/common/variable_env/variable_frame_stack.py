import random
from collections import deque, defaultdict
from itertools import count
from typing import Dict, Deque
from unittest.mock import Mock, MagicMock

import gym
import numpy as np
import pytest

from .variable_env import VariableEnv
from .variable_step_result import VariableStepResult
from .variable_wrapper import VariableWrapper


class VariableFrameStack(VariableWrapper):
    def __init__(self, env: VariableEnv, k=4):
        super().__init__(env)
        self.k = k
        self._frames: Dict[int, Deque[np.ndarray]] = defaultdict(lambda: deque([], maxlen=k))
        sp: gym.spaces.Box = env.observation_space
        self.observation_space = gym.spaces.Box(low=sp.low.repeat(k, 0), high=sp.high.repeat(k, 0),
                                                dtype=env.observation_space.dtype)

    def reset(self) -> VariableStepResult:
        res = self.env.reset()
        self._frames.clear()
        self._set_ob(res)
        return res

    def step(self, action: np.ndarray) -> VariableStepResult:
        res = self.env.step(action)
        self._set_ob(res)
        return res

    def _set_ob(self, res: VariableStepResult):
        obs = []
        for i, aid in enumerate(res.agent_id):
            stack = self._frames[aid]
            added = False
            while len(stack) < self.k or not added:
                stack.append(res.obs[i])
                added = True
            obs.append(np.concatenate(stack, axis=-len(self.observation_space.shape)))

        res.obs = np.stack(obs, 0)

        for aid, done in enumerate(res.done):
            if done and aid in self._frames:
                del self._frames[aid]


@pytest.mark.parametrize('shape, slow, shigh, dtype, k, batch', [
    [(3, 16, 16), 0, 255, np.uint8, 3, 1],
    [(1, 16, 16), 0, 255, np.uint8, 4, 4],
    [(100,), -1, 1, np.float32, 5, 7],
])
def test_VariableFrameStack_reset(shape, slow, shigh, dtype, k, batch):
    random.seed(123)
    np.random.seed(123)

    shape_stack = batch, k * shape[0], *shape[1:]
    env = MagicMock()
    env.observation_space = gym.spaces.Box(0, 255, shape=shape, dtype=dtype)
    env.reset.side_effect = env.step.side_effect = (Mock(
        obs=np.full((batch, *shape), i, dtype=dtype),
        agent_id=np.arange(batch),
        done=np.random.random(batch) < 0.2
    ) for i in count())

    stack = VariableFrameStack(env, k)

    assert len(stack._frames) == 0
    assert np.allclose(stack.reset().obs, np.full(shape_stack, 0))


@pytest.mark.parametrize('shape, slow, shigh, dtype, k, batch', [
    [(3, 16, 16), 0, 255, np.uint8, 3, 1],
    [(1, 16, 16), 0, 255, np.uint8, 4, 4],
    [(100,), -1, 1, np.float32, 5, 7],
])
def test_VariableFrameStack(shape, slow, shigh, dtype, k, batch):
    random.seed(123)
    np.random.seed(123)

    shape_stack = batch, k * shape[0], *shape[1:]
    env = MagicMock()
    env.observation_space = gym.spaces.Box(0, 255, shape=shape, dtype=dtype)

    t = 0
    def get_ob(*_):
        nonlocal t
        t += 1
        return Mock(obs=np.zeros((batch, *shape), dtype=dtype) + t - 1, agent_id=np.arange(batch), done=np.random.random(batch) < 0.2)

    env.reset.side_effect = env.step.side_effect = get_ob
    stack = VariableFrameStack(env, k=k)
    for _ in range(5):
        assert stack.reset().obs.shape == shape_stack
        for _ in range(10):
            assert stack.step(None).obs.shape == shape_stack
            assert stack.step(None).obs.reshape(-1)[0].sum() < stack.step(None).obs.reshape(-1)[-1].sum()
