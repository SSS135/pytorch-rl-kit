import copy
import itertools
import os
import pprint
import random
import subprocess
from collections import namedtuple
from functools import partial
from multiprocessing.pool import ThreadPool
from typing import Dict

from sklearn.model_selection import ParameterGrid
from torch.multiprocessing import Pool
from pathlib import Path

from . import EnvTrainer


def save_git_diff(save_folder, tag, repo_path):
    diff = subprocess.check_output(['git', 'diff'], cwd=repo_path)
    with open(f'{save_folder}/{Path(repo_path).name}_{tag}.diff', 'wb') as file:
        file.write(diff)


def rl_alg_test(hyper_params: Dict[str, list] or list, wrap_params: dict, alg_class: type, alg_params: dict, env_factory,
                num_processes: int, frames: int, iters: int=1, use_worker_id: bool=False, shuffle=False, use_threads=False) -> list:
    """
    Used for hyperparameter search and testing of reinforcement learning algorithms.
    Args:
        hyper_params: Each element contains list of possible values for specific hyperparameter.
            Contents mixed together using `ParameterGrid` and passed to RL algorithm constructor.
            Contents of `hyper_params` will overwrite contents of `alg_params`.
        wrap_params: Arguments passed to `EnvTrainer` constructor
        alg_class: Type of RL algorithm. Algorithm must be inherited from `RLBase`.
        alg_params: Arguments passed to RL algorithm constructor.
            Will be overwritten by `hyper_params` with same name.
        env_factory: Used to instantiate environment.
            Usually gym env name, but could be anything accepted by `EnvTrainer`.
        num_processes: Number of processes for hyperparameter search.
            Won't use multiprocessing, if `num_processes` equals 1 or single combination of hyperparameters is used .
        frames: Training steps.
        iters: Number of iterations for each hyperparameter combination.
        use_worker_id: Used by some environments, like ones created using Unity ML Agents.
        shuffle: shuffle run order
        use_threads: use ThreadPool instead of Pool

    Returns: list of `EnvTrainer` outputs

    """
    lowpriority()
    hyper_params = list(ParameterGrid(hyper_params)) if isinstance(hyper_params, dict) else hyper_params
    input = zip(hyper_params,
                itertools.repeat(wrap_params) if isinstance(wrap_params, dict) else wrap_params,
                itertools.repeat(alg_class),
                itertools.repeat(alg_params),
                itertools.repeat(env_factory),
                itertools.repeat(frames),
                itertools.repeat(use_worker_id),
                itertools.count())
    input = [SimInput(*x) for x in input]
    outputs = []
    for _ in range(iters):
        if shuffle:
            random.shuffle(input)
        if num_processes > 1 and len(input) > 1:
            with (ThreadPool if use_threads else Pool)(num_processes) as pool:
                outputs.extend(pool.map(simulate, input))
        else:
            for x in input:
                outputs.append(simulate(x))
    return outputs


def is_running_on_windows():
    import sys
    try:
        sys.getwindowsversion()
    except AttributeError:
        return False
    else:
        return True


def lowpriority():
    """
    Set the priority of the process to below-normal.
    https://stackoverflow.com/questions/1023038/change-process-priority-in-python-cross-platform
    """
    is_windows = is_running_on_windows()

    if is_windows:
        # Based on:
        #   "Recipe 496767: Set Process Priority In Windows" on ActiveState
        #   http://code.activestate.com/recipes/496767/
        import win32api, win32process, win32con

        pid = win32api.GetCurrentProcessId()
        handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pid)
        win32process.SetPriorityClass(handle, win32process.BELOW_NORMAL_PRIORITY_CLASS)
    else:
        import os

        os.nice(1)


SimInput = namedtuple('SimInput', 'hyper_params, wrap_params, alg_class, alg_params, '
                                  'env_factory, frames, use_worker_id, worker_id')


def simulate(input: SimInput):
    if 'tag' in input.wrap_params and is_running_on_windows():
        os.system(f'title {input.wrap_params["tag"]}')
    input = copy.deepcopy(input)
    input.alg_params.update(input.hyper_params)
    env_factory = partial(input.env_factory, worker_id=input.worker_id) if input.use_worker_id else input.env_factory
    rl_alg_factory = partial(input.alg_class, **input.alg_params)
    input.wrap_params['rl_alg_factory'] = rl_alg_factory
    input.wrap_params['env_factory'] = env_factory
    input.wrap_params['alg_name'] = input.alg_class.__name__
    gym_wrap = EnvTrainer(**input.wrap_params)
    gym_wrap.logger.add_text('hparams', pprint.pformat(input.hyper_params))
    gym_wrap.logger.add_text('wrap_params', pprint.pformat(input.wrap_params))
    gym_wrap.logger.add_text('alg_params', pprint.pformat(input.alg_params))
    return gym_wrap.train(input.frames)
