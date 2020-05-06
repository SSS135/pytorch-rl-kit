# from .actors import FCActor, Actor
# from .cnn_actors import CNNActor, Sega_CNNActor
from .heads import ActionValueHead, MultiActionValueHead, HeadBase, PolicyHead, StateValueHead
from .norm_factory import NormFactory, LambdaNormFactory, BatchNormFactory, GroupNormFactory, InstanceNormFactory, \
    LayerNormFactory
# from .qrnn_actors import CNN_QRNNActor
# from .rnn_actors import RNNActor
# from .hrnn_actors import HRNNActor
from .cnn_actors import CNNFeatureExtractor, Sega_CNNFeatureExtractor, create_ppo_cnn_actor

from .actors import Actor, ModularActor, FeatureExtractorBase
from .fc_actors import create_ppo_fc_actor, FCFeatureExtractor, create_sac_fc_actor
from .rnn_actors import create_ppo_rnn_actor
from .silu import SiLU
