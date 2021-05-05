import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

from .quantizable_layer import QuantizableModule
from .qtensor import QTensor

import copy
from typing import Dict

# wrapped _BatchNorm interface like in the recommendation of RLisfun:
# https://github.com/pytorch/pytorch/issues/4741

# compatibility of BN with model.eval() (seems defunct) via:
# 1. keeping bn.training == True always
# 2. changing bn.track_running_stats to False during eval()
# -> wenn bn.track_running_stats False, setze kurz momentum = 0, sodass kein update passiert

class _QBatchNorm(QuantizableModule, _BatchNorm):
    def __init__(self, *args, qkwargs: Dict = None, **kwargs):
        QuantizableModule.__init__(self, **qkwargs)
        _BatchNorm.__init__(self, *args, **kwargs)

    def forward_quantized(x: QTensor) -> QTensor:
        """
        # NOTES for stuff to do inside qat_convert.py:
        # rewrite/delete convbn.train und setze BN permanent auf track_running_stats=False
        # quantisierung:
        # 1. weight /= sigma; bias *= mu
        # 2. weight und bias beide auf 32 quantizen

        Quantized batch norm self, without folding

        :param x: QTensor, quantized input data
        """

        # TODO figure out how to factor out these scales properly
        # so I can assign these tensors permanently
        # (this falls under optimization tho and would not appease mr knuth)

        gamma = quantization_weight.quantize_to_qtensor(
            self.weight,
            num_bits=num_bits_weight # int8 or int32 ? TODO test
        )

        numerator_scale = x.scale * gamma.scale
        denominator_scale = numerator_scale # ** 2
        scale_out = math.sqrt(numerator_scale) # siehe overleaf rechnung
        # scale_out = numerator_scale

        mu = quant_weight.quantize_to_qtensor_given_scale(
            self.running_mean,
            x.scale,
            0,
            num_bits=num_bits_bias
        )

        sigma =  quant_weight.quantize_to_qtensor_given_scale(
            self.running_var,
            denominator_scale,
            0,
            num_bits=num_bits_weight
        )

        epsilon = quant_weight.quantize_to_qtensor_given_scale(
            torch.Tensor([self.eps]),
            denominator_scale,
            0,
            num_bits=num_bits_weight
        )

        if self.bias is not None:

            # TODO if block removen? bn sollte immer bias haben
            beta = quant_weight.quantize_to_qtensor_given_scale(
                self.bias,
                scale_out,
                0,
                num_bits=num_bits_bias
            ) # as in prep

            beta = beta._t
        else:
            beta = None

        assert self.training, "batchnorm must have .training==True always, see top of batchnorm.py"
        assert not self.track_running_stats, "see batchnorm.py"

        _, exponential_average_factor = self.do_checks(x._t)

        r = F.batch_norm(
            input=x._t - x.zero,
            running_mean=mu._t,
            running_var=sigma._t,
            weight=gamma._t - gamma.zero,
            bias=beta,
            training=bn_training,
            momentum=exponential_average_factor,
            eps=epsilon._t.item(),
        )

        multiplier = scale_out / self.scale_next

        out = r * multiplier + self.zero_next

        out = quant_input.tensor_clamp(out, num_bits=num_bits_input)

        assert is_integer(out), out

        assert len(torch.unique(out)) > 1, (out.mean(), tnsr_stats(x, quant_input), multiplier, zero_point_next, out_before)

        return QTensor(out, scale=self.scale_next, zero=self.zero_next)

    def do_checks(self, input: Tensor) -> Tensor:
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

        return bn_training, exponential_average_factor

    def forward_fp(self, input: Tensor) -> Tensor:
        bn_training, exponential_average_factor = self.do_checks(input)

        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps
        )

    def forward_qat(self, input: Tensor) -> Tensor:
        bn_training, exponential_average_factor = self.do_checks(input)

        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps
        )

    def train(self, mode:bool):

        # batchnorm can apparently not handle eval mode
        # (https://github.com/pytorch/pytorch/issues/4741)
        # instead, set following flag and use it in BatchNormWrap.forward:

        self.track_running_stats = bool(mode)

        return self

class QBatchNorm1dTranspose(_QBatchNorm, nn.BatchNorm1d):
    def forward_fp(self, x):
        x = torch.transpose(x,1,2)
        x = super().forward_fp(x)
        x = torch.transpose(x,1,2)
        return x

    def forward_qat(self, x):
        x = torch.transpose(x,1,2)
        x = super().forward_qat(x)
        x = torch.transpose(x,1,2)
        return x

    def forward_quantize(self, x):
        x = torch.transpose(x,1,2)
        x = super().forward_quantize(x)
        x = torch.transpose(x,1,2)
        return x

class QBatchNorm1d(_QBatchNorm, nn.BatchNorm1d):
    pass

class QBatchNorm2d(_QBatchNorm, nn.BatchNorm2d):
    pass

class QBatchNorm3d(_QBatchNorm, nn.BatchNorm3d):
    pass

