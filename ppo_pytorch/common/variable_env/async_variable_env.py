import time
from enum import Enum
from multiprocessing.connection import Connection
from multiprocessing import Process, Pipe
from typing import List, NamedTuple, Any, Optional, Callable

import numpy as np

from .variable_env import VariableEnv
from .variable_env_merger import VariableEnvMerger
from .variable_step_result import VariableStepResult


class Command(Enum):
    shutdown = 0
    stats = 1
    reset = 3
    step = 2
    render = 4


class Message(NamedTuple):
    command: Command
    payload: Any


def process_entry(pipe: Connection, env_fn):
    env = env_fn()
    env.reset()
    while True:
        msg: Message = pipe.recv()
        if msg.command == Command.stats:
            pipe.send((env.observation_space, env.action_space, env.env_name))
        elif msg.command == Command.reset:
            pipe.send(env.reset())
        elif msg.command == Command.step:
            pipe.send(env.step(msg.payload))
        elif msg.command == Command.render:
            env.render()
        elif msg.command == Command.shutdown:
            env.close()
            pipe.send(True)
            return


class AsyncVariableEnv(VariableEnv):
    def __init__(self, env_factories: List[Callable], min_ready_envs=0.5):
        self.min_ready_envs = max(1, round(len(env_factories) * min_ready_envs))
        self._merger = VariableEnvMerger()
        self._waiting_for_actions = []

        self._pipes = []
        self._processes = []
        for env_factory in env_factories:
            parent_conn, child_conn = Pipe()
            self._pipes.append(parent_conn)
            proc = Process(target=process_entry, args=(child_conn, env_factory))
            proc.start()
            self._processes.append(proc)
        self.observation_space, self.action_space, self.env_name = self._sync_rpc(Command.stats, pipes=[self._pipes[0]])[0]

    @property
    def num_envs(self):
        return len(self._processes)

    def step(self, action: np.ndarray) -> VariableStepResult:
        self._submit_actions(action)
        while not self._states_ready:
            time.sleep(0)
        return self._get_states()

    def reset(self) -> VariableStepResult:
        for pipe in self._waiting_for_actions:
            pipe.recv()
        self._waiting_for_actions.clear()
        self._waiting_for_actions.extend(self._pipes)
        data = self._sync_rpc(Command.reset)
        return self._merger.merge(data, list(range(len(data))))

    def render(self):
        self._pipes[0].send(Message(Command.render, None))

    def close(self):
        self._sync_rpc(Command.shutdown)
        for proc in self._processes:
            proc.terminate()
            proc.join(30)
            proc.close()

    @property
    def _states_ready(self):
        return sum(p.poll() for p in self._pipes) >= self.min_ready_envs

    def _submit_actions(self, actions: np.ndarray) -> None:
        actions = self._merger.split_actions(actions)
        assert len(self._waiting_for_actions) == len(actions)
        for pipe, ac in zip(self._waiting_for_actions, actions):
            pipe.send(Message(Command.step, ac))
        self._waiting_for_actions.clear()

    def _get_states(self) -> VariableStepResult:
        assert len(self._waiting_for_actions) == 0
        data, ids = [], []
        for i, pipe in enumerate(self._pipes):
            if pipe.poll():
                self._waiting_for_actions.append(pipe)
                data.append(pipe.recv())
                ids.append(i)
        assert len(data) > 0
        return self._merger.merge(data, ids)

    def _sync_rpc(self, command: Command, payload: Optional[List] = None, pipes: List[Connection] = None) -> List:
        if pipes is None:
            pipes = self._pipes
        if payload is None:
            payload = len(pipes) * [None]
        for pipe, payload in zip(pipes, payload):
            pipe.send(Message(command, payload))
        return [pipe.recv() for pipe in pipes]