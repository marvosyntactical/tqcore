import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from .qtensor import QTensor
from .quantization_functions import Quantization
from .batchnorm import *
from .quantizable_layer import NonQuantizableModuleWrap

import math

from torch.nn.modules.utils import _pair

import copy
from typing import Optional

MOMENTUM = .01
__DEBUG__ = False

# helper fns
printdbg = lambda *expr: printdbg(*expr) if __DEBUG__ else None
tnsr_stats = lambda t, qinp: (round(t.min().item(), 3), round(t.max().item(), 3),qinp.calc_zero(t.min().item(), t.max().item(), 8))
is_integer = lambda t: (t.round()==t).all() if __DEBUG__ else True



# contains factory functions for quantized forward passes used in .qat_convert.py
# for pytorch layers such as linear, conv, batchnorm


# ============================= PYTORCH OPS ==================================

def _factory_convert_layer_forward_impl(module):

    # op-for-op implementation of quantized layers given range params a, b

    # TODO quantize weights and everything here!!!! not in the forward functions below (on the fly...)

    q_layer_fwd = _factory_quantized_layer(module)

    if isinstance(module, nn.Conv2d):
        settings = {
            "stride": module.stride,
            "padding": module.padding,
            "dilation": module.dilation,
            "groups": module.groups,
        }
    elif isinstance(module, nn.Linear):
        settings = {}
    elif issubclass(type(module), nn.modules.batchnorm._BatchNorm):
        settings = {
            "training": True, # needs to be True because of issues mentioned in .batchnorm.py
            "momentum": module.momentum,
        }
    elif isinstance(module, nn.MaxPool2d):
        settings = {}
    else:
        raise ValueError(f"Found layer of type {type(module)} which isnt in list of implemented module types")

    def _converted_layer_forward_impl(x, *args, **kwargs) -> Tensor:
        """
        Wrapper function for layer fwd pass
        min_val and max_val are fixed and set to the averages recorded during QAT
        """
        x_q = q_layer_fwd(
            x=x,
            layer=module,
            quant_input=module._Qinp,
            quant_weight=module._Qwt,
            min_val=module.__stats__["ema_min"],
            max_val=module.__stats__["ema_max"],
            num_bits_input=module._num_bits_inp,
            num_bits_weight=module._num_bits_wt,
            num_bits_bias=module._num_bits_bias if hasattr(module, "_num_bits_bias") else None,
            **settings
        )

        return x_q # torch.Tensor
    return _converted_layer_forward_impl

def _factory_quantized_layer(module:nn.Module):

    def non_quantized_batchnorm(
            x:QTensor,
            layer:torch.nn.modules.batchnorm._BatchNorm,
            quant_input:Quantization,
            quant_weight:Quantization,
            min_val:int,
            max_val:int,
            num_bits_weight=8,
            num_bits_input=8,
            num_bits_bias=32,
            **kwargs):

        # NOTES for stuff to do inside qat_convert.py:
        # rewrite/delete convbn.train und setze BN permanent auf track_running_stats=False
        # quantisierung:
        # 1. weight /= sigma; bias *= mu
        # 2. weight und bias beide auf 32 quantizen

        """
        Quantized batch norm layer, without folding

        :param x: quantized input data
        :param layer: the batch norm layer to be quantized
        :param quant_input: Quantization function for the activations
        :param quant_weight: Quantization function for the weights
        :param min_val: EMA min_val for next activation
        :param max_val: EMA max_val for next activation
        :param num_bits_input: bit width of input
        :param num_bits_weight: bit width of weight
        :return: QTensor: (output, output_scale, output_zero)
        """

        # here: ganzzahliges x rescalen:

        assert layer.training, "batchnorm must have .training==True always, see batchnorm.py"
        assert not layer.track_running_stats, "see batchnorm.py"

        x_float_rescaled = x.dequantize()

        out = F.batch_norm(
            input=x_float_rescaled,
            weight=layer.weight,
            bias=layer.bias,
            running_mean=layer.running_mean,
            running_var=layer.running_var,
            eps=layer.eps,
            **kwargs
        )

        out = out / layer.scale_next + layer.zero_next

        out_before_clamp = tnsr_stats(out, quant_input)

        out = quant_input.tensor_clamp(out, num_bits=num_bits_input)

        assert is_integer(out), out

        assert len(torch.unique(out)) > 1, (
            out.mean(),
            tnsr_stats(x, quant_input),
            scale_next,
            zero_next,
            out_before_clamp
        )

        return QTensor(out, scale=layer.scale_next, zero=layer.zero_next)

    def quantized_linear(
            x:torch.Tensor,
            layer:torch.nn.Linear,
            quant_input:Quantization,
            quant_weight:Quantization,
            min_val:int,
            max_val:int,
            num_bits_input=8,
            num_bits_weight=8,
            num_bits_bias=32,
            **kwargs):
        """
        Quantized linear layer, functionality according to https://arxiv.org/pdf/1712.05877.pdf, section 2.
        For fully quantized inference, input tensor has to be quantized either by quantizing initial input of model or is quantized
        output of previous quantized layer.

        :param x: quantized input data
        :param layer: the current torch layer
        :param quant_input: Quantization function for the activations
        :param quant_weight: Quantization function for the weights
        :param min_val: EMA min_val for next activation
        :param max_val: EMA max_val for next activation
        :param num_bits_input: bit width of input
        :param num_bits_weight: bit width of weight
        :return: QTensor: (output, output_scale, output_zero)
        """
        w = quant_weight.quantize_to_qtensor(
            layer.weight.data,
            num_bits=num_bits_weight
        )

        if layer.bias is not None:
            b = quant_weight.quantize_to_qtensor_given_scale(
                layer.bias.data,
                w.scale * x.scale,
                0,
                num_bits=num_bits_bias
            )
            b = b._t
        else:
            b = None

        x_zeroed = x._t - x.zero
        w_zeroed = w._t - w.zero

        out = F.linear(x_zeroed,  w_zeroed, bias=b, **kwargs)

        multiplier = (x.scale * w.scale) / layer.scale_next
        # scale result tensor back to given bit width, saturate to uint if unsigned is used
        out = out * multiplier + layer.zero_next

        out = quant_input.tensor_clamp(out, num_bits=num_bits_input)

        return QTensor(out, scale=layer.scale_next, zero=layer.zero_next)

    def quantized_conv2d(
            x:torch.Tensor,
            layer:torch.nn.Conv2d,
            quant_input:Quantization,
            quant_weight:Quantization,
            min_val,
            max_val,
            num_bits_input=8,
            num_bits_weight=8,
            num_bits_bias=32,
            **kwargs
            ):
        """
        Quantized convolutional layer, functionality according to https://arxiv.org/pdf/1712.05877.pdf, section 2.
        For fully quantized inference, input tensor has to be quantized either by quantizing initial input of model or is quantized
        output of previous quantized layer.

        :param x: quantized input data
        :param layer: the current torch layer
        :param quant_input: Quantization function for the activations
        :param quant_weight: Quantization function for the weights
        :param min_val: EMA min_val for next activation
        :param max_val: EMA max_val for next activation
        :param num_bits_input: bit width of input
        :param num_bits_weight: bit width of weight
        :return: QTensor: (output, output_scale, output_zero)
        """

        w = quant_weight.quantize_to_qtensor(
            layer.weight.data,
            num_bits=num_bits_weight
        )

        if layer.bias is not None:
            b = quant_weight.quantize_to_qtensor_given_scale(
                layer.bias.data,
                w.scale * x.scale,
                0,
                num_bits=num_bits_bias,
            )
            b = b._t
        else:
            b = None

        w_zeroed = w._t - w.zero
        x_zeroed = x._t - x.zero

        pad = layer._reversed_padding_repeated_twice if hasattr(layer, "_reversed_padding_repeated_twice") else tuple(x for x in reversed(layer.padding) for _ in range(2))

        out = F.conv2d(
            F.pad(
                x_zeroed,
                pad,
                mode="constant",
                value=0,
            ),
            w_zeroed,
            b,
            layer.stride,
            _pair(0),
            layer.dilation,
            layer.groups,
        )

        multiplier = ( x.scale * w.scale ) / layer.scale_next

        # scale result tensor back to given bit width, saturate to uint if unsigned is used
        out = out * multiplier + layer.zero_next

        out = quant_input.tensor_clamp(out, num_bits=num_bits_input)

        # Update activation tensor quantization values

        return QTensor(out, scale=layer.scale_next, zero=layer.zero_next)

    def quantized_maxpool2d(
            x:torch.Tensor,
            layer:torch.nn.Conv2d,
            quant_input:Quantization,
            quant_weight:Quantization,
            min_val,
            max_val,
            num_bits_input=8,
            num_bits_weight=8,
            num_bits_bias=32,
            **kwargs
            ):
        """
        Quantized convolutional layer, functionality according to https://arxiv.org/pdf/1712.05877.pdf, section 2.
        For fully quantized inference, input tensor has to be quantized either by quantizing initial input of model or is quantized
        output of previous quantized layer.

        :param x: quantized input data
        :param layer: the current torch layer
        :param quant_input: Quantization function for the activations
        :param quant_weight: Quantization function for the weights
        :param min_val: EMA min_val for next activation
        :param max_val: EMA max_val for next activation
        :param num_bits_input: bit width of input
        :param num_bits_weight: bit width of weight
        :return: QTensor: (output, output_scale, output_zero)
        """

        x_zeroed = x._t - x.zero

        pad = (layer.padding,)*4

        out = F.max_pool2d(
            F.pad(
                x_zeroed,
                pad,
                mode="constant",
                value=0,
            ),
            layer.kernel_size,
            stride=layer.stride,
            padding=0,
            dilation=layer.dilation,
        )

        multiplier = x.scale / scale_next

        # scale result tensor back to given bit width, saturate to uint if unsigned is used
        out = out * multiplier + zero_next

        out = quant_input.tensor_clamp(out, num_bits=num_bits_input)

        assert is_integer(out), out

        # Update activation tensor quantization values

        return QTensor(out, scale=layer.scale_next, zero=layer.zero_next)

    quantized_ops = {
        nn.BatchNorm1d: non_quantized_batchnorm,
        nn.Linear: quantized_linear,
        nn.Conv2d: quantized_conv2d,
        nn.MaxPool2d: quantized_maxpool2d,
    }

    fun = quantized_ops.get(type(module), False)
    assert fun, type(module)
    return fun

global __OPS__, __NONQUANT__
__OPS__ = [
    nn.Conv2d,
    nn.Linear,
    nn.modules.batchnorm._BatchNorm,
    nn.MaxPool2d
]

# ignore subclasses of these:
__NONQUANT__ = [
    NonQuantizableModuleWrap
]



