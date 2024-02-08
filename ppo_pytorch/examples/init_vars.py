# %load_ext snakeviz

import platform

is_linux = platform.system() == 'Linux'

import torch
import ppo_pytorch as rl
import gymnasium as gym
import gymnasium.spaces
import math
import numpy as np
import numpy.random as rng
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from functools import partial
from torch.optim.rmsprop import RMSprop
from ppo_pytorch.common.rl_alg_test import rl_alg_test
import ppo_pytorch.common
import ppo_pytorch.actors
import ppo_pytorch.algs
from ppo_pytorch.common.utils import save_git_diff
import time


log_path = '/mnt/c/Users/Alexander/sync-pc/Jupyter/tensorboard' if is_linux else './tensorboard'
diff_path = '/mnt/c/Users/Alexander/sync-pc/Jupyter/diffs' if is_linux else './diffs'
src_root = '/mnt/c/Users/Alexander/sync-pc/PyCharm' if is_linux else 'c:/Users/Alexander/sync-pc/PyCharm'

time_str = time.strftime("%Y-%m-%d %H_%M_%S")
save_git_diff(diff_path, time_str, f'{src_root}/ppo_pytorch')
save_git_diff(diff_path, time_str, f'{src_root}/rl_exp')
save_git_diff(diff_path, time_str, f'{src_root}/optfn')
save_git_diff(diff_path, time_str, f'{src_root}/raic2019')
print()
print(time_str, '\n')

import __main__
with open(__main__.__file__, 'r') as script:
    print(script.read())