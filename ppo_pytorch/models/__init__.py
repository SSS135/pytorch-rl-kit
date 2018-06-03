from functools import partial

from .actors import MLPActor, CNNActor, Actor, Sega_CNNActor
from .qrnn_actors import QRNNActor, CNN_QRNNActor, Sega_CNN_HQRNNActor, HQRNNActor
from .seq_actors import Sega_CNNSeqActor
from .heads import ActionValuesHead, ActorCriticHead, HeadBase

MLPActorCritic = partial(MLPActor, head_factory=ActorCriticHead)
CNNActorCritic = partial(CNNActor, head_factory=ActorCriticHead)
QRNNActorCritic = partial(QRNNActor, head_factory=ActorCriticHead)
HQRNNActorCritic = partial(HQRNNActor, head_factory=ActorCriticHead)
CNN_QRNNActorCritic = partial(CNN_QRNNActor, head_factory=ActorCriticHead)
# Sega_CNN_QRNNActorCritic = partial(Sega_CNN_QRNNActor, head_factory=ActorCriticHead)
Sega_CNNActorCritic = partial(Sega_CNNActor, head_factory=ActorCriticHead)
Sega_CNNSeqActorCritic = partial(Sega_CNNSeqActor, head_factory=ActorCriticHead)
Sega_CNN_HQRNNActorCritic = partial(Sega_CNN_HQRNNActor, head_factory=ActorCriticHead)

MLPActionValues = partial(MLPActor, head_factory=ActionValuesHead)
CNNActionValues = partial(CNNActor, head_factory=ActionValuesHead)
