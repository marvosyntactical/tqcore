import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

from .quantizable_layer import QuantizableModule, SYMMETRIZING_MODULES
from .kernel import qmul, qadd
from .qtensor import QTensor
from .quantization_functions import FakeQuant

import copy
from typing import Dict, Optional, Union
import math
import warnings

__DEBUG__ = True
is_integer = lambda t: (t.round()==t).all() if __DEBUG__ else True

# wrapped _BatchNorm interface like in the recommendation of RLisfun:
# https://github.com/pytorch/pytorch/issues/4741

# compatibility of BN with model.eval() (seems defunct) via:
# 1. keeping bn.training == True always
# 2. changing bn.track_running_stats to False during eval()
# -> wenn bn.track_running_stats False, setze kurz momentum = 0, sodass kein update passiert

class _QBatchNorm(QuantizableModule, _BatchNorm):

    def __init__(self, *args, qkwargs: Dict, **kwargs):
        QuantizableModule.__init__(self, **qkwargs)
        _BatchNorm.__init__(self, *args, **kwargs)
        self.record_n_batches = qkwargs["record_n_batches_bn"]

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
        assert x.num_bits == self.num_bits, (x.num_bits, self.num_bits)

        # TODO assign these tensors permanently during .quantize() ....
        # (this falls under optimization tho and would not appease mr knuth)

        gamma = self.weight_quantization.quantize_to_qtensor_using_range(
            self.folded_weight,
            num_bits=self.num_bits_weight # int8 or int32 ? TODO test
        )

        beta = self.quantization.quantize_to_qtensor_given_scale(
            self.folded_bias,
            x.scale * gamma.scale,
            0,
            num_bits=32,
        )

        # TODO replace below code by quantized_layer.quantized_linear call
        # / abstract with replaceable mul kernel (matmul/mul)

        # assert not self.track_running_stats, "<-- this attr indicates whether we are training; forward_quantized is for inference only"

        x_zeroed = x._t - x.zero
        gamma_zeroed = gamma._t - gamma.zero

        out = x_zeroed * gamma_zeroed + beta._t

        multiplier = (x.scale * gamma.scale) / self.scale

        out = out * multiplier + self.zero

        out = self.quantization.tensor_clamp(out, num_bits=self.num_bits)

        out = QTensor(out, scale=self.scale, zero=self.zero, num_bits=self.num_bits, quantized=True)

        assert is_integer(out._t), out

        assert len(torch.unique(out._t)) > 1, (out.min(), x.min(), x.max(), self.zero, self.scale)

        return out # QTensor(out, scale=self.scale, zero=self.zero)

    def _do_checks(self, input: Tensor) -> Tensor:
        # NOTE uncomment TODO
        # self._check_input_dim(input)

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
        bn_training, exponential_average_factor = self._do_checks(input)

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

    def forward_qat(self, input: QTensor) -> QTensor:
        assert input.num_bits == self.num_bits, (input.num_bits, self.num_bits)
        super().forward_qat()
        bn_training, exponential_average_factor = self._do_checks(input)

        out = F.batch_norm(
            input._t,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps
        )
        if self.n_qat_batches == self.record_n_batches:
            self.freeze()
        return QTensor(out, scale=self.scale, zero=self.zero, num_bits=self.num_bits, quantized=False)

    def freeze(self):
        # to stop recording running stats;
        # make self never trainable using self.train() again.
        print("="*30)
        print(self, " stopped recording.")
        print("="*30)
        self.track_running_stats = True
        self.train = lambda t: warnings.warn(f"{self}.train({t}) got called after it was frozen!"); return self

    def quantize(self):
        QuantizableModule.quantize(self)

        eps = torch.sqrt(torch.Tensor([self.eps]))

        inv_running_std = 1/torch.sqrt(self.running_var + eps)
        self.folded_weight = (self.weight * inv_running_std) \
                .unsqueeze(0).unsqueeze(-1)

        self.folded_bias = self.bias.unsqueeze(0).unsqueeze(-1) \
                - (self.running_mean.unsqueeze(0).unsqueeze(-1) \
                * self.folded_weight)

    def train(self, mode:bool):
        # batchnorm can apparently not handle eval mode
        # (https://github.com/pytorch/pytorch/issues/4741)
        # instead, set following flag and use it in BatchNormWrap.forward:
        self.track_running_stats = bool(mode)
        return self

class QBatchNorm1dTranspose(_QBatchNorm, nn.BatchNorm1d):
    def __init__(self, *args, qkwargs: Dict, **kwargs):
        _QBatchNorm.__init__(self, *args, qkwargs=qkwargs, **kwargs)
        nn.BatchNorm1d.__init__(self, *args, **kwargs)

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

class FPBatchNorm1dTranspose(nn.BatchNorm1d):
    def __init__(self, *args, **kwargs):
        nn.BatchNorm1d.__init__(self, *args, **kwargs)

    def forward(self, x):
        x = torch.transpose(x,1,2)
        x = nn.BatchNorm1d.forward(self, x)
        x = torch.transpose(x,1,2)
        return x

class QBatchNorm1d(_QBatchNorm, nn.BatchNorm1d):
    pass

class QBatchNorm2d(_QBatchNorm, nn.BatchNorm2d):
    pass

class QBatchNorm3d(_QBatchNorm, nn.BatchNorm3d):
    pass

class ConvBNfoldable(QuantizableModule):
    """
    relu(batchnorm2d(conv2d(x))) style module
    with custom forward pass that is altered during qat_prepare and qat_convert

    structure from https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html#quantization-aware-training
    as described in https://arxiv.org/abs/1712.05877v1 Sec 3.2
    This module switches to the Fig C8 procedure upon call of self.qat_prepare():
    https://bluemountain.eee.hku.hk/papaa2018/PAPAA18-L04-Jac+18.pdf
    Then folds/removes the BN completely when self.fold() is called
    """
    def __init__(
            self,
            in_planes,
            out_planes,
            kernel_size, #=3,
            stride=1,
            groups=1,
            padding=0,
            momentum=0.1,
            relu:Union[int, bool]=6,
            eps=1e-05,
            Convclass=nn.Conv2d,
            **qkwargs
        ):

        super().__init__(**qkwargs)

        if not type(padding) == int:
            # dont know what this is for; its from https://pytorch.org/tutorials/advanced/static_quantization_tutorial.htm
            padding = (kernel_size - 1) // 2

        self.conv = Convclass(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = QBatchNorm2d(out_planes, momentum=momentum)

        self.has_relu = not(type(relu) == bool and relu == False)

        if self.has_relu:
            relu_module = nn.ReLU() if (type(relu)==int and relu==0) else nn.ReLU6()
            self.relu = relu_module

        def debug_fold_backward(module, grad_in, grad_out):
            # sanity check if weights have been updated since last backward pass
            convweight = module.conv.weight.data
            bnweight = module.bn.weight.data
            bnbias = module.bn.bias.data
            if hasattr(module, "last_weights_cache"):
                # are these ever updated?
                if not (convweight == module.last_weights_cache[0]).all():
                    print("conv weight updated!")
                if not (bnweight == module.last_weights_cache[1]).all():
                    print("bn weight updated!")
                if not (bnbias == module.last_weights_cache[2]).all():
                    print("bn bias updated")

            module.last_weights_cache = [convweight]
            module.last_weights_cache += [bnweight]
            module.last_weights_cache += [bnbias]

        # self.register_backward_hook(
        #        debug_fold_backward
        #        )

    def forward_fp(self, x):
        # forward used during fp32 pretraining
        assert not (not self.conv.training and self.bn.track_running_stats)

        x = self.conv(x)
        x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)
        return x

    def forward_quantized(self, x):
        # forward used after conversion, no bn
        x = self.conv(x)
        if self.has_relu:
            x = self.relu(x)
        return x

    def quantize(self):
        super().quantize()
        self.fold()

    def folded_weight(self):
        # C8: w_fold = w * (gamma/sigma)
        folded_weight = self.conv.weight * (self.bn.weight / torch.sqrt(self.bn.running_var + self.bn.eps)).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        return folded_weight

    def folded_bias(self):
        # C8: bias = beta - gamma * mu / sigma
        folded_bias = self.bn.bias -  (self.bn.weight * self.bn.running_mean) / torch.sqrt(self.bn.running_var + self.bn.eps)
        return folded_bias

    def fold(self):

        folded_weight = self.folded_weight()
        folded_bias = self.folded_bias()

        assert not torch.isnan(folded_weight).any()
        assert not torch.isnan(folded_bias).any()

        self.conv.weight.data = folded_weight
        if self.conv.bias is not None:
            self.conv.bias.data = folded_bias
        else:
            self.conv.bias = nn.Parameter(folded_bias)

        # # change function to normal fwd pass again, but wthout bn
        # self.forward = self.forward_folded


    def forward_qat(self, x):
        """
        https://bluemountain.eee.hku.hk/papaa2018/PAPAA18-L04-Jac+18.pdf Fig C8 procedure
        """
        assert not (not self.conv.training and self.bn.track_running_stats)

        if self.bn.track_running_stats:
            # update BN running stats
            self.bn(self.conv(x))

        # fold the weights then fake quantize in conv's fwd hook:

        # unterer Teil des training graphs in C8:

        folded_weight = self.folded_weight()
        folded_weight.data = self.conv._fakeQ(folded_weight.data, self.conv._Qwt, self.conv._num_bits_wt, None, None, handling_qtensors=False)

        folded_bias = self.folded_bias()
        folded_bias.data = self.conv._fakeQ(folded_bias.data, self.conv._Qwt, self.conv._num_bits_bias, None, None, handling_qtensors=False)

        assert not torch.isnan(folded_weight).any()
        assert not torch.isnan(folded_bias).any()

        # DEBUG: try adding bias with functional batch norm instead ..?

        # oberer teil des training graphs in C8:
        x = F.conv2d(
            F.pad(
                x,
                self.conv._reversed_padding_repeated_twice if hasattr(self.conv, "_reversed_padding_repeated_twice") else tuple(x_ for x_ in reversed(self.conv.padding) for _ in range(2)),
            ),
            folded_weight,
            folded_bias,
            self.conv.stride,
            _pair(0),
            self.conv.dilation,
            self.conv.groups,
        )

        if self.has_relu:
            x = self.relu(x)
        return x

class QBNFoldableTranspose(QuantizableModule):
    """
    # FIXME for the moment only works with QAT TODO implement calibration
    container module with custom forward pass thats altered during qat_prepare and qat_convert

    via https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html#quantization-aware-training
    as described in https://arxiv.org/abs/1712.05877v1 Sec 3.2

    This module
    Switches the Fig C8 procedure upon call of self.qat_prepare():
    https://bluemountain.eee.hku.hk/papaa2018/PAPAA18-L04-Jac+18.pdf
    """
    def __init__(
            self,
            hidden,
            momentum=0.1,
            eps=1e-05,
            dimension: int=1, # bn1d, bn2d, bn3d
            **qkwargs
        ):

        super().__init__(**qkwargs)

        self.bn = nn.BatchNorm1d(hidden, momentum=momentum, eps=eps)
        self._fakeQ = FakeQuant.apply_wrapper

    def qat_prepare(self, **qkwargs):
        super().qat_prepare(**qkwargs)
        if 0:
            def debug_fold_backward(module, grad_in, grad_out):
                # sanity check if weights have been updated since last backward pass
                bnweight = module.bn.weight.data
                bnbias = module.bn.bias.data
                if hasattr(module, "last_weights_cache"):
                    # are these ever updated?
                    if not (bnweight == module.last_weights_cache[0]).all():
                        print("bn weight updated!")
                    if not (bnbias == module.last_weights_cache[1]).all():
                        print("bn bias updated")

                module.last_weights_cache = [bnweight]
                module.last_weights_cache += [bnbias]

            self.hook = self.register_backward_hook(
                debug_fold_backward
            )


    def forward_quantized(self, x: QTensor) -> QTensor:
        x = torch.transpose(x, 1, 2)

        gamma = self.weight_quantization.quantize_to_qtensor_using_range(
            self.folded_weight,
            num_bits=self.num_bits_weight # int8 or int32 ? TODO test
        )

        beta = self.quantization.quantize_to_qtensor_given_scale(
            self.folded_bias,
            x.scale * gamma.scale,
            0,
            num_bits=32,
        )

        x_zeroed = x._t - x.zero
        gamma_zeroed = gamma._t - gamma.zero

        out = x_zeroed * gamma_zeroed + beta._t

        multiplier = (x.scale * gamma.scale) / self.scale
        out = out * multiplier + self.zero

        out = self.quantization.tensor_clamp(out, num_bits=self.num_bits)

        out = QTensor(out, scale=self.scale, zero=self.zero)

        assert is_integer(out._t), out

        assert len(torch.unique(out._t)) > 1, (out.min(), x.min(), x.max(), self.zero, self.scale)

        return torch.transpose(out, 1,2) # QTensor(out, scale=self.scale, zero=self.zero)

    def forward_qat(self, x):
        """
        https://bluemountain.eee.hku.hk/papaa2018/PAPAA18-L04-Jac+18.pdf Fig C8 procedure without Conv
        """

        x = torch.transpose(x, 1, 2)

        if self.bn.track_running_stats:
            # training =>
            # update BN running stats
            self.bn(x)

        # fold the weights then fake quantize:

        # unterer Teil des training graphs in C8:

        folded_weight = self.fold_weight()
        folded_weight.data = self._fakeQ(folded_weight.data, self.weight_quantization, self.num_bits_weight, None, None, handling_qtensors=False)

        folded_bias = self.fold_bias(folded_weight)
        folded_bias.data = self._fakeQ(folded_bias.data, self.weight_quantization, self.num_bits_weight, None, None, handling_qtensors=False)

        x = x._t * folded_weight + folded_bias

        x = torch.transpose(x, 1, 2)

        return QTensor(x, scale=self.scale, zero=self.zero, quantized=False)

    def forward_fp(self, x):
        # forward used during fp32 pretraining
        x = torch.transpose(x, 1, 2)
        x = self.bn(x)
        x = torch.transpose(x, 1, 2)
        return x

    def quantize(self):
        super().quantize()
        self.fold()
        if hasattr(self, "hook"):
            self.hook.remove()
            del self.hook

    def fold_weight(self):

        inv_running_std = 1/torch.sqrt(self.bn.running_var + self.bn.eps)
        folded_weight = (self.bn.weight * inv_running_std)\
                .unsqueeze(0).unsqueeze(-1) # same layout as input
        return folded_weight

    def fold_bias(self, folded_weight):
        folded_bias = self.bn.bias.unsqueeze(0).unsqueeze(-1) \
                - (self.bn.running_mean.unsqueeze(0).unsqueeze(-1) \
                * folded_weight)
        return folded_bias

    def fold(self):
        self.folded_weight = self.fold_weight()
        self.folded_bias = self.fold_bias(self.folded_weight)

SYMMETRIZING_MODULES += [
    QBatchNorm1dTranspose,
    QBNFoldableTranspose,
]
