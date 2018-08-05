from collections import namedtuple

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from .ppo import PPO, TrainingData
from ..models import QRNNActor
from ..models.heads import HeadOutput
from ..models.utils import image_to_float
from ..models.heads import PolicyHead, StateValueHead


class SAC(PPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def head_factory(hidden_size, pd):
        return dict(probs=PolicyHead(hidden_size, pd), state_value=StateValueHead(hidden_size),
                    action_value=StateValueHead(hidden_size))


