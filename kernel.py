import torch
from torch import Tensor
from torch.nn import Parameter

from typing import Union, Callable, Tuple

from .qtensor import QTensor
from .quantization_functions import Quantization
from .utils import print_qt_stats, is_integer

# This module contains "kernels" that simulate low bit addition, multiplication, and matmul
# TODO future:
# replace qadd and qmul by gemmlowp (possibly <) 8 bit kernel

def _convert_to_qtensor_for_kernel(
        a,
        b,
        quantization,
        weight_quantization,
        num_bits,
        num_bits_weight,
        quantize=True,
    ) -> Tuple[torch.Tensor]:
    # helper function

    ab = [a,b]
    for i, t in enumerate(ab):
        if not isinstance(t, QTensor):
            if isinstance(t, Tensor):
                if isinstance(t, Parameter):
                    # e.g. elementwise mul with parameter
                    t = weight_quantization.quantize_to_qtensor_using_range(
                        t,
                        min_val=t.min().item(),
                        max_val=t.max().item(),
                        num_bits=num_bits_weight,
                        quantized=quantize
                    )
                else:
                    # e.g. rescale in mhattn
                    t = quantization.quantize_to_qtensor_using_range(
                        t,
                        min_val=t.min().item(),
                        max_val=t.max().item(),
                        num_bits=num_bits,
                        quantized=quantize
                    )
            else:
                assert False, (t, type(t))
                # t = QTensor(torch.as_tensor(t), scale=1., zero=0.)
        ab[i] = t
    return tuple(ab)

def qadd(
        a: Union[Tensor, QTensor, Parameter],
        b: Union[Tensor, QTensor, Parameter],
        factor: float,
        scale_next: float,
        zero_next: float,
        op: Callable,
        quantization: Quantization,
        weight_quantization: Quantization,
        num_bits: int,
        num_bits_weight: int
    ) -> QTensor:
    # mock low bit addition kernel

    # tensor wrapper version of the earlier "globalparams" implementation:
    # https://cegit.ziti.uni-heidelberg.de/mkoss/tqcore/-/blob/globalparams/quantized_layer.py#L206

    a, b = _convert_to_qtensor_for_kernel(
        a, b, quantization, weight_quantization, num_bits, num_bits_weight
    )

    denom = a.scale + b.scale
    a_frac = a.scale / denom
    b_frac = b.scale / denom

    a_dq = (a._t - a.zero) * a.scale
    b_dq = (b._t - b.zero) * b.scale

    a_rq = a_dq / scale_next + zero_next
    b_rq = b_dq / scale_next + zero_next

    a_rq *= a_frac
    b_rq *= b_frac

    # NOTE:
    # for accurate simulation of quantization, it is crucial that round() and clamp() happen
    # before the tensors get added
    bits = (num_bits-1)
    a_rq = a_rq.round().clamp(0, (2.**bits)-1.)
    b_rq = b_rq.round().clamp(0, (2.**bits)-1.)

    r = a_rq + b_rq

    # r = a_rq + b_rq
    # r = r.round()

    # a_requantized = quantization.quantize_to_qtensor_using_params(
    #     a.dequantize() * factor,
    #     scale=scale_next*.5,
    #     zero=zero_next,
    #     num_bits=num_bits # half the scale
    # )
    # print_qt_stats("qadd a", a_requantized)
    # b_requantized = quantization.quantize_to_qtensor_using_params(
    #     b.dequantize() * factor,
    #     scale=scale_next*.5,
    #     zero=zero_next,
    #     num_bits=num_bits # half the scale
    # )
    # print_qt_stats("qadd b", b_requantized)

    # r = a_requantized._t + b_requantized._t
    # r = r.clamp(0., (2.**num_bits)-1.)
    r = QTensor(r, scale=scale_next, zero=zero_next)

    # print_qt_stats("qadd result", r)

    assert is_integer(r._t), r

    return r

def qmul(
        a: Union[Tensor, QTensor, Parameter],
        b: Union[Tensor, QTensor, Parameter],
        factor: float, # for scaled matmul as in MHAttn (/sqrt(key))
        scale_next: float,
        zero_next: float,
        op: Callable,
        quantization: Quantization,
        weight_quantization: Quantization,
        num_bits: int,
        num_bits_weight: int,
        quantize=True,
    ) -> QTensor:
    # mock low bitwidth kernel for mul and matmul

    a, b = _convert_to_qtensor_for_kernel(
        a, b, quantization, weight_quantization, num_bits, num_bits_weight
    )

    a_zeroed = a._t - round(a.zero)
    b_zeroed = b._t - round(b.zero)

    r: torch.Tensor = op(a_zeroed, b_zeroed)
    r_float = r

    multiplier = (a.scale * b.scale * factor) / scale_next
    # scale result tensor back to given bit width, saturate to uint if unsigned is used:
    r = r * multiplier + zero_next
    r_unclamped = r

    # round and clamp
    r = quantization.tensor_clamp(r, num_bits=num_bits)

    # Make sure we didnt move outside of EMA range:
    assert r.min() != r.max(), \
        (scale_next, zero_next, r.min(), a._t.min(), a._t.max(), b._t.min(), b._t.max(), r_float.min(), r_float.max(), r_unclamped.min(), r_unclamped.max(), multiplier, factor)

    return QTensor(r, scale=scale_next, zero=zero_next, quantized=quantize)


