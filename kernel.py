import torch
from torch import Tensor
from torch.nn import Parameter

from typing import Union, Callable, Tuple
import warnings
import math

from .qtensor import QTensor
from .quantization_functions import Quantization, quant_logger

is_integer = lambda t: t.allclose(t.round())

# This module contains "kernels" that simulate low bit addition, multiplication, and matmul
# (the terminology "kernel" is from https://github.com/google/gemmlowp : low bit compute functions)
# TODO future:
# replace qadd and qmul by gemmlowp like (possibly <) 8 bit kernel
# (gemmlowp also contains hints on how to do this for less than 8 bits)

def _convert_to_qtensor_for_kernel(
        a,
        b,
        quant_a,
        quant_b,
        num_bits_a,
        num_bits_b,
        quantized=True,
    ) -> Tuple[torch.Tensor]:
    # helper function

    ab = [a,b]
    for i, (t, Q, nb) in enumerate(zip(ab, [quant_a, quant_b], [num_bits_a, num_bits_b])):
        if not isinstance(t, QTensor):
            if isinstance(t, Tensor):
                if isinstance(t, Parameter):
                    # e.g. elementwise mul with parameter
                    t = Q.quantize_to_qtensor_using_range(
                        t,
                        min_val=t.min().item(),
                        max_val=t.max().item(),
                        num_bits=nb,
                        quantized=quantized
                    )
                else:
                    assert False
                    # # e.g. rescale in mhattn
                    # t = Q.quantize_to_qtensor_using_range(
                    #     t,
                    #     min_val=t.min().item(),
                    #     max_val=t.max().item(),
                    #     num_bits=nb,
                    #     quantized=quantized
                    # )
            else:
                assert False, (t, type(t))
                # t = QTensor(torch.as_tensor(t), scale=1., zero=0.)
        ab[i] = t
    return tuple(ab)

def qadd(
        a: Union[Tensor, QTensor, Parameter],
        b: Union[Tensor, QTensor, Parameter],
        scale_next: float,
        zero_next: float,
        quant_a: Quantization,
        quant_b: Quantization,
        quant_out: Quantization,
        num_bits_a: int,
        num_bits_b: int,
        num_bits_out: int,
        op: Callable = torch.add,
        rescale=True
    ) -> QTensor:
    # mock low bit addition kernel

    # NOTE TODO
    # assumes both a and b are activations; or that weight quantization is the same
    # as activation quantization at least

    a, b = _convert_to_qtensor_for_kernel(
        a, b, quant_a, quant_b, num_bits_a, num_bits_b
    )
    assert a.num_bits==num_bits_a
    assert b.num_bits==num_bits_b

    denom = a.scale + b.scale

    a_dq = (a._t - a.zero) * a.scale
    b_dq = (b._t - b.zero) * b.scale

    a_rq = a_dq / scale_next + zero_next
    b_rq = b_dq / scale_next + zero_next

    # # # TODO derive this rescaling
    a_rq *= a.scale / denom
    b_rq *= b.scale / denom

    # NOTE:
    # for accurate simulation of quantization, it is crucial that round() and clamp() happen
    # before the tensors get added:

    # a_rq = a_rq.round().clamp(0, (2.**num_bits_out)-1.)
    a_rq = quant_a.tensor_clamp(a_rq, num_bits_a)

    # b_rq = b_rq.round().clamp(0, (2.**num_bits_out)-1.)
    b_rq = quant_b.tensor_clamp(b_rq, num_bits_b)

    # NOTE perform addition
    r = op(a_rq, b_rq)

    scale=scale_next
    zero=zero_next

    if rescale:
        # rescale if necessary:
        qmax = (2.**num_bits_out)-1.
        if (r > qmax).any():
            # lin et al 2020 "towards fully 8-bit integer inference for the transformer model" sec 3.3
            re_scale = r.max().item() / qmax
            r = r / re_scale
            # r = r.round().clamp(0, (2.**num_bits_out)-1.)
            r = quant_out.tensor_clamp(r, num_bits_out)

            # Q = (R / S + Z) / re_scale
            # =>

            # NOTE adjust parameters after scaling
            zero = zero / re_scale
            scale = scale * re_scale

    r = QTensor(r, scale=scale, zero=zero, num_bits=num_bits_out, quantized=True)

    assert is_integer(r._t), r

    return r

def qmul(
        a: Union[Tensor, QTensor, Parameter],
        b: Union[Tensor, QTensor, Parameter],
        factor: float, # for scaled matmul as in MHAttn (/sqrt(key))
        scale_next: float,
        zero_next: float,
        op: Callable,
        quant_a: Quantization,
        quant_b: Quantization,
        quant_out: Quantization,
        num_bits_a: int,
        num_bits_b: int,
        num_bits_out: int,
    ) -> QTensor:
    # mock low bitwidth kernel for mul and matmul

    a, b = _convert_to_qtensor_for_kernel(
        a, b, quant_a, quant_b, num_bits_a, num_bits_b
    )
    assert a.num_bits==num_bits_a
    assert b.num_bits==num_bits_b

    a_zeroed = a._t - round(a.zero)
    b_zeroed = b._t - round(b.zero)

    r: torch.Tensor = op(a_zeroed, b_zeroed)
    r_float = r

    multiplier = (a.scale * b.scale * factor) / scale_next
    # scale result tensor back to given bit width, saturate to uint if unsigned is used:
    r = r * multiplier + zero_next
    r_unclamped = r

    # round and clamp
    r = quant_out.tensor_clamp(r, num_bits=num_bits_out)

    # DEBUG: Make sure we didnt move outside of EMA range:
    if r.min() == r.max():
        expressions = ["scale_next", "zero_next", "r.min()", "a._t.min()", "a._t.max()", "b._t.min()", "b._t.max()", "r_float.min()", "r_float.max()", "r_unclamped.min()", "r_unclamped.max()", "multiplier", "factor"]
        msg = "\n"
        for expression in expressions:
            msg += expression+ "\t = \t"+ str(eval(expression)) + "\n"
        warnings.warn(msg)

    assert is_integer(r), r

    return QTensor(r, scale=scale_next, zero=zero_next, num_bits=num_bits_out, quantized=True)


