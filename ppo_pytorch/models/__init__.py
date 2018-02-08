from functools import partial

from .actors import MLPActor, CNNActor
from .heads import ActionValuesHead, ActorCriticHead, HeadBase

MLPActorCritic = partial(MLPActor, head_factory=ActorCriticHead)
CNNActorCritic = partial(CNNActor, head_factory=ActorCriticHead)
MLPActionValues = partial(MLPActor, head_factory=ActionValuesHead)
CNNActionValues = partial(CNNActor, head_factory=ActionValuesHead)
