# from .actors import FCActor, Actor
# from .cnn_actors import CNNActor, Sega_CNNActor
from .heads import ActionValuesHead, HeadBase, PolicyHead, StateValueHead
from .norm_factory import NormFactory, LambdaNormFactory, BatchNormFactory, GroupNormFactory, InstanceNormFactory, \
    LayerNormFactory
# from .qrnn_actors import CNN_QRNNActor
# from .rnn_actors import RNNActor
# from .hrnn_actors import HRNNActor
from .cnn_actors import CNNFeatureExtractor, Sega_CNNFeatureExtractor, create_ppo_cnn_actor

from .actors import Actor, ModularActor, FeatureExtractorBase
from .fc_actors import create_ppo_fc_actor, FCFeatureExtractor, create_td3_fc_actor
