from functools import partial

from .actors import MLPActor, CNNActor, Actor, ActorOutput, Sega_CNNActor
from .qrnn_actors import QRNNActor, CNN_QRNNActor, Sega_CNN_QRNNActor
from .heads import ActionValuesHead, ActorCriticHead, HeadBase

MLPActorCritic = partial(MLPActor, head_factory=ActorCriticHead)
CNNActorCritic = partial(CNNActor, head_factory=ActorCriticHead)
QRNNActorCritic = partial(QRNNActor, head_factory=ActorCriticHead)
CNN_QRNNActorCritic = partial(CNN_QRNNActor, head_factory=ActorCriticHead)
Sega_CNN_QRNNActorCritic = partial(Sega_CNN_QRNNActor, head_factory=ActorCriticHead)
Sega_CNNActorCritic = partial(Sega_CNNActor, head_factory=ActorCriticHead)

MLPActionValues = partial(MLPActor, head_factory=ActionValuesHead)
CNNActionValues = partial(CNNActor, head_factory=ActionValuesHead)
