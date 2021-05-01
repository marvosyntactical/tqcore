import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from .qtensor import QTensor
from .quantization_functions import Quantization
from .quantizable_layer import *
from .batchnorm import *

import math

from torch.nn.modules.utils import _pair

import copy
from typing import Optional

MOMENTUM = .01
__DEBUG__ = False

# helper fns
printdbg = lambda *expr: printdbg(*expr) if __DEBUG__ else None
tnsr_stats = lambda t, qinp: (round(t.min().item(), 3), round(t.max().item(), 3),qinp.calc_zero_point(t.min().item(), t.max().item(), 8))
qparams_stats = lambda qparams_dict: tuple(map(lambda fl: round(fl, 3), qparams_dict.values()))
is_integer = lambda t: (t.round()==t).all()

# contains factory functions for quantized forward passes used in .qat_convert.py:
# separate factories for activations
# one big factory for most other layers

# ============================= DEBUG ==================================

def _debug_forward_post_hook(mod, inp, out):
    t = out[0]
    assert (t==t.round()).all(), (type(mod), t)
    assert len(torch.unique(out)) > 1, (torch.unique(out), type(mod))

# ============================= PYTORCH ACTIVS ==================================

def _factory_convert_relu_layer_forward_impl(module, min_val, max_val, num_bits_inp):
    def _converted_relu_layer_forward_impl(x, *args, **kwargs):

        assert is_integer(x)

        zero = module.__qparams__["zero_point"]
        out = x.clamp(min=zero)

        # FIXME for debug only:

        scale = module.__qparams__["scale"]

        actual_scale, actual_zero = module._Qinp.calc_zero_point(
                min_val, max_val, num_bits_inp)

        assert round(scale, 5) == round(actual_scale, 5), \
                (scale, actual_scale)
        assert round(zero, 5) == round(actual_zero, 5), \
                (zero, actual_zero)

        return out
    return _converted_relu_layer_forward_impl


def _factory_convert_relu6_layer_forward_impl(module, min_val, max_val, num_bits_inp):
    def _converted_relu6_layer_forward_impl(x, *args, **kwargs):

        assert is_integer(x), x

        scale = module.__qparams__["scale"]
        zero = module.__qparams__["zero_point"]
        six = round(6 / scale + zero)

        actual_scale, actual_zero = module._Qinp.calc_zero_point(
            min_val,
            max_val,
            num_bits_inp
        )

        assert round(scale, 5) == round(actual_scale, 5), \
                (scale, actual_scale)
        assert round(zero, 5) == round(actual_zero, 5), \
                (zero, actual_zero)

        out = x.clamp(min=zero, max=six)

        return out
    return _converted_relu6_layer_forward_impl

# ============================= PYTORCH LAYERS ==================================

def _factory_convert_layer_forward_impl(module, min_val, max_val):

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
    elif isinstance(module, QuantizableResConnection):
        settings = {}
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
            min_val=min_val,
            max_val=max_val,
            num_bits_input=module._num_bits_inp,
            num_bits_weight=module._num_bits_wt,
            num_bits_bias=module._num_bits_bias if hasattr(module, "_num_bits_bias") else None,
            qparams=module.__qparams__,
            **settings
        )

        return x_q # torch.Tensor
    return _converted_layer_forward_impl

def _factory_quantized_layer(module:nn.Module):

    def quantized_res_connection(
            x:torch.Tensor,
            layer:QuantizableResConnection,
            quant_input:Quantization,
            min_val,
            max_val,
            num_bits_input=8,
            qparams=None,
            **kwargs):
        """
        Rescale older tensor to scale of newer one
        given by qparams dict (last updated to params of new tensor)

        :param x: older tensor that gets added to newer other tensor
        :param layer: the current torch layer
        :param quant_input: Quantization function for the activations
        :param min_val: EMA min_val for next activation
        :param max_val: EMA max_val for next activation
        :param num_bits_input: bit width of input
        :param qparams: qparams for other, more recent, newer tensor (not old)
        :return: QTensor: (output, output_scale, output_zero_point)
        """
        old = x

        assert is_integer(old)

        # FIXME testing different rescaling here:
        # instead of halfing both newer and older scale, adjust each relatively

        fraction_of_next_scale = .5

        M = layer.old_scale / qparams["scale"]

        out = (old - layer.old_zero) * M
        out = out + qparams["zero_point"]
        out *= fraction_of_next_scale

        printdbg("quantized residual: older tensor (arg 0) rounding error:")
        printdbg(f"avg rounding err: {(out-out.round()).abs().mean()}")
        out = quant_input.tensor_clamp(out, num_bits=num_bits_input-1)

        return out

    def quantized_batchnorm(
            x:torch.Tensor,
            layer:torch.nn.modules.batchnorm._BatchNorm,
            quant_input:Quantization,
            quant_weight:Quantization,
            min_val:int,
            max_val:int,
            num_bits_weight=8,
            num_bits_input=8,
            num_bits_bias=32,
            qparams=None,
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
        :return: QTensor: (output, output_scale, output_zero_point)
        """

        # here: ganzzahliges x rescalen:

        assert layer.training, "batchnorm must have .training==True always, see batchnorm.py"
        assert not layer.track_running_stats, "see batchnorm.py"

        assert is_integer(x), \
            (tnsr_stats(x, layer._Qinp), qparams)

        x_float_rescaled = qparams["scale"] * ( x - qparams["zero_point"])

        # assert False, \
        #         "stats: {}".format("\n".join([str(tnsr_stats(tnsr)) for tnsr in \
        #         (layer.weight, layer.bias, x, x_float_rescaled) \
        #         ]))

        out = F.batch_norm(
            input=x_float_rescaled, # x_float_rescaled,
            weight=layer.weight,
            bias=layer.bias,
            running_mean=layer.running_mean,
            running_var=layer.running_var,
            eps=layer.eps,
            **kwargs
        )

        scale_next, zero_point_next = quant_input.calc_zero_point(
            min_val=min_val, # ema for lower bound of next activation
            max_val=max_val, # ema for upper bound of next activation
            num_bits=num_bits_input
        )

        out = out / scale_next + zero_point_next

        # should already be integer after this ^
        # but isnt quite, TODO get standard deviation

        out_before_clamp = tnsr_stats(out, quant_input)

        printdbg("non quantized BN: older tensor (arg 0) rounding error:")
        printdbg(f"avg rounding err: {(out-out.round()).abs().mean()}")
        out = quant_input.tensor_clamp(out, num_bits=num_bits_input)

        # uncomment!
        assert is_integer(out), out

        assert len(torch.unique(out)) > 1, (out.mean(), tnsr_stats(x, quant_input),scale_next, zero_point_next, out_before_clamp)

        # printdbg(tnsr_stats(out), zero_point_next)

        qparams["scale"]=scale_next
        qparams["zero_point"]=zero_point_next

        return out

    def quantized_batchnorm_nofold(x:torch.Tensor, layer:torch.nn.BatchNorm2d, quant_input:Quantization, quant_weight:Quantization,  min_val:int, max_val:int, num_bits_weight=8, num_bits_input=8, num_bits_bias=32,qparams=None, **kwargs):

        """
        # NOTES for stuff to do inside qat_convert.py:
        # rewrite/delete convbn.train und setze BN permanent auf track_running_stats=False
        # quantisierung:
        # 1. weight /= sigma; bias *= mu
        # 2. weight und bias beide auf 32 quantizen

        Quantized batch norm layer, without folding

        :param x: quantized input data
        :param layer: the batch norm layer to be quantized
        :param quant_input: Quantization function for the activations
        :param quant_weight: Quantization function for the weights
        :param min_val: EMA min_val for next activation
        :param max_val: EMA max_val for next activation
        :param num_bits_input: bit width of input
        :param num_bits_weight: bit width of weight
        :return: QTensor: (output, output_scale, output_zero_point)
        """
        assert False, "unused and out of date"

        # FIXME finde heraus wie man ohne reinfalten mu, sigma quantisieren könnte

        # ============ attempt at quantization without folding =================

        assert layer.training, "batchnorm must have .training==True always, see batchnorm.py"
        assert not layer.track_running_stats, "see batchnorm.py"

        gamma = quant_weight.quantize_to_qtensor(
                layer.weight,
                num_bits=num_bits_bias
        )
        numerator_scale = qparams["scale"] * gamma.scale
        denominator_scale = numerator_scale # ** 2
        scale_out = math.sqrt(numerator_scale)
        # scale_out = numerator_scale

        mu = quant_weight.quantize_to_qtensor_given_scale(
                layer.running_mean,
                qparams["scale"],
                0,
                num_bits=num_bits_bias)


        sigma =  quant_weight.quantize_to_qtensor_given_scale(
                layer.running_var,
                denominator_scale,
                0,
                num_bits=num_bits_bias)

        epsilon = quant_weight.quantize_to_qtensor_given_scale(
                torch.Tensor([layer.eps]),
                denominator_scale,
                0,
                num_bits=num_bits_bias)

        assert is_integer(x), \
                (tnsr_stats(x, quant_weight), qparams)
        assert is_integer(gamma.tensor), \
                (gamma, tnsr_stats(gamma.tensor, quant_weight))
        assert is_integer(mu.tensor), \
                (mu, tnsr_stats(mu.tensor, quant_weight))
        assert is_integer(sigma.tensor), \
                (sigma, tnsr_stats(sigma.tensor, quant_weight))

        if layer.bias is not None:

            # TODO if block removen? bn sollte immer bias haben
            beta = quant_weight.quantize_to_qtensor_given_scale(
                    layer.bias,
                    scale_out,
                    0,
                    num_bits=num_bits_bias) # as in prep

            beta = beta.tensor

            assert is_integer(beta), \
                    (beta, tnsr_stats(beta, quant_weight))
        else:
            beta = None

        # ============================= end attempt =============================

        out = F.batch_norm(
                input=x - qparams["zero_point"],
                weight=gamma.tensor - gamma.zero_point,
                bias=beta,
                running_mean=mu.tensor,
                running_var=sigma.tensor,
                eps=epsilon.tensor.item(),
                **kwargs
        )

        scale_next, zero_point_next = quant_input.calc_zero_point(
                min_val=min_val,
                max_val=max_val,
                num_bits=num_bits_input
        )

        multiplier = scale_out / scale_next

        out = out * multiplier + zero_point_next
        out_before = tnsr_stats(out, quant_input)

        # should already be integer after this ^
        # but isnt quite, TODO get std

        printdbg(f"non folded BN rounding error:")
        printdbg(f"avg rounding err: {(out-out.round()).abs().mean()}")
        out = quant_input.tensor_clamp(out, num_bits=num_bits_input)

        # uncomment!
        assert is_integer(out), out

        assert len(torch.unique(out)) > 1, (out.mean(), tnsr_stats(x, quant_input), multiplier, zero_point_next, out_before)

        # printdbg(tnsr_stats(out), zero_point_next)

        qparams["scale"]=scale_next
        qparams["zero_point"]=zero_point_next

        return out

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
            qparams=None,
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
        :return: QTensor: (output, output_scale, output_zero_point)
        """
        w = quant_weight.quantize_to_qtensor(
            layer.weight.data,
            num_bits=num_bits_weight
        )

        if not is_integer(x):
            # TODO put this functionality in something like a QuantStub class
            printdbg(f"input to quantized {type(layer)} # {layer._module_number} not integer, make sure this is intended (happens after avg pool / mean operations)")
            printdbg("Input range:", tnsr_stats(x, quant_input))
            printdbg("rounding.")
            x = x.round()

        assert is_integer(w.tensor), (w, tnsr_stats(w.tensor, quant_input))

        if layer.bias is not None:
            b = quant_weight.quantize_to_qtensor_given_scale(
                layer.bias.data,
                w.scale * qparams["scale"],
                0,
                num_bits=num_bits_bias
            )

        x_zeroed = x - qparams["zero_point"]
        w_zeroed = w.tensor - w.zero_point

        if layer.bias is not None:
            out = F.linear(x_zeroed,  w_zeroed, bias=b.tensor, **kwargs)
        else:
            out = F.linear(x_zeroed,  w_zeroed)

        scale_next, zero_point_next = quant_input.calc_zero_point(
            min_val=min_val,
            max_val=max_val,
            num_bits=num_bits_input
        )

        multiplier = (qparams["scale"] * w.scale) / scale_next
        # scale result tensor back to given bit width, saturate to uint if unsigned is used
        out = out * multiplier + zero_point_next
        # should already be integer after this ^
        # but isnt quite, TODO get std

        printdbg(f"quantized linear rounding error:")
        printdbg(f"avg rounding err: {(out-out.round()).abs().mean()}")
        out = quant_input.tensor_clamp(out, num_bits=num_bits_input)
        assert is_integer(out), out

        # Update activation tensor quantization values
        qparams["scale"]=scale_next
        qparams["zero_point"]=zero_point_next

        return out

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
            qparams=None,
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
        :return: QTensor: (output, output_scale, output_zero_point)
        """

        w = quant_weight.quantize_to_qtensor(
            layer.weight.data,
            num_bits=num_bits_weight
        )

        if layer.bias is not None:
            b = quant_weight.quantize_to_qtensor_given_scale(
                layer.bias.data,
                w.scale * qparams["scale"],
                0,
                num_bits=num_bits_bias,
            )

        w_zeroed = w.tensor - w.zero_point
        x_zeroed = x - qparams["zero_point"]

        pad = layer._reversed_padding_repeated_twice if hasattr(layer, "_reversed_padding_repeated_twice") else tuple(x for x in reversed(layer.padding) for _ in range(2))

        out = F.conv2d(
            F.pad(
                x_zeroed,
                pad,
                mode="constant",
                value=0,
            ),
            w_zeroed,
            b.tensor if layer.bias is not None else None,
            layer.stride,
            _pair(0),
            layer.dilation,
            layer.groups,
            # **kwargs
        )

        scale_next, zero_point_next = quant_input.calc_zero_point(
            min_val=min_val,
            max_val=max_val,
            num_bits=num_bits_input
        )

        multiplier = ( qparams["scale"] * w.scale ) / scale_next

        # scale result tensor back to given bit width, saturate to uint if unsigned is used
        out = out * multiplier + zero_point_next

        printdbg("quantized conv2d rounding error:")
        printdbg(f"avg rounding err: {(out-out.round()).abs().mean()}")
        out = quant_input.tensor_clamp(out, num_bits=num_bits_input)

        assert is_integer(out), out

        # Update activation tensor quantization values
        qparams["scale"] = scale_next
        qparams["zero_point"] = zero_point_next

        return out

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
            qparams=None,
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
        :return: QTensor: (output, output_scale, output_zero_point)
        """

        x_zeroed = x - qparams["zero_point"]

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

        scale_next, zero_point_next = quant_input.calc_zero_point(
            min_val=min_val,
            max_val=max_val,
            num_bits=num_bits_input
        )

        multiplier = ( qparams["scale"] ) / scale_next

        # scale result tensor back to given bit width, saturate to uint if unsigned is used
        out = out * multiplier + zero_point_next

        printdbg("maxpool2d rounding error:")
        printdbg(f"avg rounding err: {(out-out.round()).abs().mean()}")
        out = quant_input.tensor_clamp(out, num_bits=num_bits_input)

        assert is_integer(out), out

        # Update activation tensor quantization values
        qparams["scale"] = scale_next
        qparams["zero_point"] = zero_point_next

        return out

    quantized_ops = {
        nn.Linear: quantized_linear,
        nn.Conv2d: quantized_conv2d,
        QuantizableResConnection: quantized_res_connection,
        BatchNorm2dWrap: quantized_batchnorm,
        BatchNorm1dWrap: quantized_batchnorm,
        nn.MaxPool2d: quantized_maxpool2d,
    }

    fun = quantized_ops.get(type(module), False)
    assert fun, type(module)
    return fun




