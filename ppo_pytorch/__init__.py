import torch
torch.set_num_threads(4)
torch.set_num_interop_threads(4)

from . import common
from . import actors
from . import algs

from .common.procgen_easy import register_environments
register_environments()