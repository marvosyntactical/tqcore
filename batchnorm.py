import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

from .quantizable_layer import QuantizableModule, _qmul, _qadd, print_qt_stats
from .qtensor import QTensor

import copy
from typing import Dict, Optional
import math

__DEBUG__ = True
is_integer = lambda t: (t.round()==t).all() if __DEBUG__ else True

# wrapped _BatchNorm interface like in the recommendation of RLisfun:
# https://github.com/pytorch/pytorch/issues/4741

# compatibility of BN with model.eval() (seems defunct) via:
# 1. keeping bn.training == True always
# 2. changing bn.track_running_stats to False during eval()
# -> wenn bn.track_running_stats False, setze kurz momentum = 0, sodass kein update passiert

class _QBatchNorm(QuantizableModule, _BatchNorm):
    def batch_norm_fun(
            self,
            input: torch.Tensor,
            running_mean: Optional[Tensor],
            running_var: Optional[Tensor],
            weight: Optional[Tensor] = None,
            bias: Optional[Tensor] = None,
            training: bool = None,
            momentum: float = 0.1,
            eps: float = 1e-5,
        ) -> Tensor:


        if self.stage == 2:

            # in low bitwidth.
            # running_var is denominator already inversed; eps can be ignored; just multiply
            running_mean = running_mean.unsqueeze(1).unsqueeze(0)
            running_var = running_var.unsqueeze(1).unsqueeze(0)

            weight = weight.unsqueeze(1).unsqueeze(0)
            bias = bias.unsqueeze(1).unsqueeze(0)
            print(f"input={input.shape}; mean={running_mean.shape}")
            centered_inp = _qadd(
                input, running_mean, 1.,
                bias.scale, 0, torch.add,
                self.quantization, self.weight_quantization,
                self.num_bits, self.num_bits_weight
            )
            print_qt_stats("centered input", centered_inp)
            print_qt_stats("denom", running_var)
            print_qt_stats("weight", weight)
            print(f"weight={weight.shape}; var={running_var.shape}")
            weightvar = _qmul(
                running_var, weight, 1.,
                self.scale_next, self.zero_next, torch.mul,
                self.quantization, self.weight_quantization,
                self.num_bits, self.num_bits_weight
            )
            print(f"weightvar={weightvar.shape}; centered_inp={centered_inp.shape}")
            lhs = _qmul(
                weightvar, centered_inp, 1.,
                self.scale_next, self.zero_next, torch.mul,
                self.quantization, self.weight_quantization,
                self.num_bits, self.num_bits_weight
            )
            print(f"lhs={lhs.shape}; bias={bias.shape}")
            r = _qadd(
                lhs, bias, 1.,
                self.scale_next, self.zero_next, torch.add,
                self.quantization, self.weight_quantization,
                self.num_bits, self.num_bits_weight
            )

        else:
            r = F.batch_norm(
                input=input,
                running_mean=running_mean,
                running_var=running_var,
                weight=weight,
                bias=bias,
                training=training,
                momentum=momentum,
                eps=eps,
            )

        return r

    def __init__(self, *args, qkwargs: Dict = None, **kwargs):
        QuantizableModule.__init__(self, **qkwargs)
        _BatchNorm.__init__(self, *args, **kwargs)

    def forward_quantized(self, x: QTensor) -> QTensor:
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

        gamma = self.weight_quantization.quantize_to_qtensor(
            self.weight,
            num_bits=self.num_bits_weight # int8 or int32 ? TODO test
        )

        numerator_scale = x.scale * gamma.scale
        scale_out = math.sqrt(numerator_scale)
        # scale_out = numerator_scale
        denominator_scale = scale_out
        scales = (numerator_scale, scale_out, denominator_scale)
        input(f"inv_running_std: {self.inv_running_std.min()} {self.inv_running_std.max()};\nscale: {denominator_scale}\nscales:{scales}")

        mu = self.weight_quantization.quantize_to_qtensor_given_scale(
            - self.running_mean,
            x.scale,
            0,
            num_bits=self.num_bits_bias
        )

        sigma =  self.weight_quantization.quantize_to_qtensor_given_scale(
            self.inv_running_std,
            denominator_scale,
            0,
            num_bits=self.num_bits_weight
        )

        epsilon = self.weight_quantization.quantize_to_qtensor_given_scale(
            self.eps,
            denominator_scale,
            0,
            num_bits=self.num_bits_weight
        )

        if self.bias is not None:
            # TODO if block removen? bn sollte immer bias haben
            beta = self.weight_quantization.quantize_to_qtensor_given_scale(
                self.bias,
                scale_out,
                0,
                num_bits=self.num_bits_bias
            ) # as in prep

            beta = beta
        else:
            beta = None

        assert not self.track_running_stats, "<-- this attr indicates whether we are training; forward_quantized is for inference only"

        x_zeroed = x._t - x.zero
        gamma_zeroed = gamma._t - gamma.zero

        bn_training, exponential_average_factor = self.do_checks(x_zeroed)

        r = self.batch_norm_fun(
            input=x,
            running_mean=mu,
            running_var=sigma,
            weight=gamma,
            bias=beta,
            training=bn_training,
            momentum=exponential_average_factor,
            eps=epsilon._t.item(),
        )

        # r = self.batch_norm_fun(
        #     input=x_zeroed,
        #     running_mean=mu._t,
        #     running_var=sigma._t,
        #     weight=gamma_zeroed,
        #     bias=beta._t,
        #     training=bn_training,
        #     momentum=exponential_average_factor,
        #     eps=epsilon._t.item(),
        # )
        # multiplier = scale_out / self.scale_next

        # out = r * multiplier + self.zero_next

        # out = self.quantization.tensor_clamp(out, num_bits=self.num_bits)

        assert is_integer(out), out

        assert len(torch.unique(out)) > 1, (out.min(), x.min(), x.max(), multiplier, self.zero_next, self.scale_next)

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

        return self.batch_norm_fun(
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

        return self.batch_norm_fun(
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

    def quantize(self):
        QuantizableModule.quantize(self)

        self.running_std = torch.sqrt(self.running_var)
        eps = self.eps
        self.eps = torch.sqrt(torch.Tensor([eps]))

        self.inv_running_std = 1/torch.sqrt(self.running_var + eps)

    def train(self, mode:bool):

        # batchnorm can apparently not handle eval mode
        # (https://github.com/pytorch/pytorch/issues/4741)
        # instead, set following flag and use it in BatchNormWrap.forward:

        self.track_running_stats = bool(mode)

        return self

class QBatchNorm1dTranspose(_QBatchNorm, nn.BatchNorm1d):
    def forward_fp(self, x):
        x = torch.transpose(x,1,2)
        x = _QBatchNorm.forward_fp(self, x)
        x = torch.transpose(x,1,2)
        return x

    def forward_qat(self, x):
        x = torch.transpose(x,1,2)
        x = _QBatchNorm.forward_qat(self, x)
        x = torch.transpose(x,1,2)
        return x

    def forward_quantized(self, x):
        x = torch.transpose(x,1,2)
        x = _QBatchNorm.forward_quantized(self, x)
        x = torch.transpose(x,1,2)
        return x

class QBatchNorm1d(_QBatchNorm, nn.BatchNorm1d):
    pass

class QBatchNorm2d(_QBatchNorm, nn.BatchNorm2d):
    pass

class QBatchNorm3d(_QBatchNorm, nn.BatchNorm3d):
    pass

