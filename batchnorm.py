import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

import copy

# wrapped _BatchNorm interface like in the recommendation of RLisfun:
# https://github.com/pytorch/pytorch/issues/4741

# compatibility of BN with model.eval() (seems defunct) via:
# 1. keeping bn.training == True always
# 2. changing bn.track_running_stats to False during eval()
# -> wenn bn.track_running_stats False, setze kurz momentum = 0, sodass kein update passiert

class _BatchNormWrap(_BatchNorm):
    def __init__(self, *args, **kwargs):
        super(_BatchNormWrap, self).__init__(*args, **kwargs)

    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore
                self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        # --------------- THIS IS THE ONLY CHANGE HERE: ---------------

        if not self.track_running_stats:
            # workaround to tell batchnorm not to update during inference
            exponential_average_factor = 0.0

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        assert self.running_mean is None or isinstance(self.running_mean, torch.Tensor)
        assert self.running_var is None or isinstance(self.running_var, torch.Tensor)
        return F.batch_norm(
                    input,
                    # If buffers are not to be tracked, ensure that they won't be updated
                    self.running_mean if not self.training or self.track_running_stats else None,
                    self.running_var if not self.training or self.track_running_stats else None,
                    self.weight,
                    self.bias,
                    bn_training,
                    exponential_average_factor,
                    self.eps)

    def train(self, mode:bool):

        # batchnorm can apparently not handle eval mode
        # (https://github.com/pytorch/pytorch/issues/4741)
        # instead, set following flag and use it in BatchNormWrap.forward:

        self.track_running_stats = bool(mode)

        return self


class BatchNorm1dWrap(_BatchNormWrap, nn.BatchNorm1d):
    pass

class BatchNorm2dWrap(_BatchNormWrap, nn.BatchNorm2d):
    pass

class BatchNorm3dWrap(_BatchNormWrap, nn.BatchNorm3d):
    pass

