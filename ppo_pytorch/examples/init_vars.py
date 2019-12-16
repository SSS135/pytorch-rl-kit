# %load_ext snakeviz

import platform
is_linux = platform.system() == 'Linux'

import torch
import ppo_pytorch as rl
import gym
import gym.spaces
import math
import numpy as np
import numpy.random as rng
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from functools import partial
from torch.optim.rmsprop import RMSprop
from ppo_pytorch.common.rl_alg_test import rl_alg_test, save_git_diff
import time


log_path = './tensorboard'
diff_path = './diffs'

time_str = time.strftime("%Y-%m-%d %H_%M_%S")
save_git_diff(diff_path, time_str, 'c:/Users/Alexander/sync-pc/PyCharm/ppo_pytorch')
save_git_diff(diff_path, time_str, 'c:/Users/Alexander/sync-pc/PyCharm/rl_exp')
save_git_diff(diff_path, time_str, 'c:/Users/Alexander/sync-pc/PyCharm/optfn')
save_git_diff(diff_path, time_str, 'c:/Users/Alexander/sync-pc/PyCharm/raic2019')
print()
print(time_str, '\n')

import __main__
with open(__main__.__file__, 'r') as script:
    print(script.read())