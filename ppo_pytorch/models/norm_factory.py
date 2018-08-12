from typing import Callable

import torch.nn as nn


class NormFactory:
    def __init__(self, allow_after_first_layer=True, allow_fc=True, allow_cnn=True, disable_bias=True):
        self.allow_after_first_layer = allow_after_first_layer
        self.allow_fc = allow_fc
        self.allow_cnn = allow_cnn
        self.disable_bias = disable_bias

    def _fc_factory(self, num_features):
        raise NotImplementedError

    def _cnn_factory(self, num_features):
        raise NotImplementedError

    def create_fc_norm(self, num_features, first_layer):
        make = self.allow_fc and (self.allow_after_first_layer or not first_layer)
        return self._fc_factory(num_features) if make else None

    def create_cnn_norm(self, num_features, first_layer):
        make = self.allow_cnn and (self.allow_after_first_layer or not first_layer)
        return self._cnn_factory(num_features) if make else None


class LambdaNormFactory(NormFactory):
    def __init__(self, fc_factory: Callable or NormFactory, cnn_factory:  Callable or NormFactory,
                 *args, **kwargs):
        super().__init__(*args, **kwargs, allow_fc=fc_factory is not None, allow_cnn=cnn_factory is not None)
        self._fc_factory = fc_factory._fc_factory if isinstance(fc_factory, NormFactory) else fc_factory
        self._cnn_factory = cnn_factory._cnn_factory if isinstance(cnn_factory, NormFactory) else cnn_factory


class GroupNormFactory(NormFactory):
    def __init__(self, group_size=16, allow_after_first_layer=False, *args, **kwargs):
        super().__init__(allow_after_first_layer, *args, **kwargs)
        self.group_size = group_size

    def _fc_factory(self, num_features):
        return nn.GroupNorm(num_features // self.group_size, num_features)

    _cnn_factory = _fc_factory


class InstanceNormFactory(NormFactory):
    def __init__(self, allow_after_first_layer=False, *args, **kwargs):
        super().__init__(allow_after_first_layer, *args, **kwargs)

    def _fc_factory(self, num_features):
        return nn.InstanceNorm1d(num_features, affine=True)

    def _cnn_factory(self, num_features):
        return nn.InstanceNorm2d(num_features, affine=True)


class LayerNormFactory(NormFactory):
    def __init__(self, allow_after_first_layer=False, *args, **kwargs):
        super().__init__(allow_after_first_layer, allow_cnn=False, *args, **kwargs)

    def _fc_factory(self, num_features):
        return nn.LayerNorm((num_features,))

    def _cnn_factory(self, num_features):
        raise NotImplementedError()


class BatchNormFactory(NormFactory):
    def _fc_factory(self, num_features):
        return nn.BatchNorm1d(num_features)

    def _cnn_factory(self, num_features):
        return nn.BatchNorm2d(num_features)