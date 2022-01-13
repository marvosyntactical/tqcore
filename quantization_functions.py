"""
Code by Lisa Kuhn, taken from https://cegit.ziti.uni-heidelberg.de/bklein/dl-frameworks-quantization/

https://arxiv.org/pdf/1712.05877.pdf
https://github.com/google/gemmlowp

Uniform and Symmetric quantizer class to quantize a tensor.

Fake quantization operator for quantization of weights
and activations during forward pass, uses STE for backward pass.

Quantized ReLU for variable zero point.

See Also here
https://github.com/eladhoffer/utils.pytorch/blob/master/quantize.py

"""

from collections import namedtuple
from typing import Tuple, Union, Dict, Optional
import math
import torch
from torch import Tensor
from .qtensor import QTensor

class Quantization():
    def __init__(self, **kwargs):
        pass

    @classmethod
    def from_opt(cls):
        return cls()

    def quantize_to_qtensor_using_range(self, x: Tensor, num_bits: int, min_val: Optional[float]=None, max_val: Optional[float]=None):
        pass

    def quantize_to_qtensor_given_scale(self, x: Tensor, scale: float, zero: float, num_bits: int) -> QTensor:
        pass

    def calc_params(self, min_val: float, max_val: float, num_bits: int):
        pass


    def tensor_clamp(self, x: Tensor, num_bits: int):
        pass



class UniformQuantization(Quantization):
    """
    Uniform affine quantization as described in https://arxiv.org/pdf/1712.05877.pdf, section 2.

    Code a la
    https://github.com/google/gemmlowp/blob/master/doc/quantization_example.cc
    """
    def __init__(self, nudge_zero=True):
        self.nudge_zero = nudge_zero

    def _quantize_tensor_using_params(self, x: Tensor, scale: float, zero: float, num_bits: int):
        q_x = x / scale + zero
        q_x = self.tensor_clamp(q_x, num_bits=num_bits)
        return q_x

    def _quantize_tensor(self, x: Tensor, num_bits: int, min_val: Optional[float]=None, max_val: Optional[float]=None) -> Tuple[Tensor, float, float]:
        # assert not num_bits==8, num_bits # NOTE debug, remove if assert

        if not min_val and not max_val:
            # weight quantization: dont actually clamp (use full range)
            min_val, max_val = x.min().item(), x.max().item()

        # scale ist real; zero is int
        scale, zero = self.calc_params(min_val=min_val, max_val=max_val, num_bits=num_bits)

        q_x = self._quantize_tensor_using_params(x=x, scale=scale, zero=zero, num_bits=num_bits)

        return q_x, scale, zero

    def quantize_to_qtensor_using_range(
            self,
            x: Tensor,
            num_bits: int,
            min_val: Optional[float]=None,
            max_val: Optional[float]=None,
            quantized: bool=True
        ) -> QTensor:

        q_x, scale, zero = self._quantize_tensor(
            x, min_val=min_val, max_val=max_val, num_bits=num_bits
        )
        return QTensor(q_x, scale=scale, zero=zero, symmetric=False, num_bits=num_bits, quantized=quantized)

    def quantize_to_qtensor_using_params(self, x: Tensor, scale: float, zero: float, num_bits: int, quantized: bool=True) -> QTensor:

        q_x = self._quantize_tensor_using_params(x, scale=scale, zero=zero, num_bits=num_bits)
        return QTensor(q_x, scale=scale, zero=zero, quantized=quantized, num_bits=num_bits, symmetric=False)

    def calc_params(self, min_val: float, max_val: float, num_bits: int, nudge: bool=True) -> Tuple[float,Union[float, int]]:
        qmin = 0.
        qmax = 2. ** num_bits - 1.

        # NOTE why is this applied here
        # max_val = max(max_val, 0.)
        # min_val = min(min_val, 0.)

        # scale ist quasi die unit länge in der quantisierten range
        scale = (max_val - min_val) / (qmax - qmin)

        # r = s * (q - z)
        initial_zero = qmin - min_val / scale

        """
        "in both cases (wt, bias), the boundaries [a; b] are nudged so that
        value 0.0 is exactly representable as an integer z(a, b, n)
        after quantization" ~ benoit et al.
        """

        if initial_zero < qmin:
            zero = qmin
        elif initial_zero > qmax:
            zero = qmax
        else:
            if self.nudge_zero or nudge:
                zero = int(initial_zero)
            else:
                zero = initial_zero

        return scale, zero

    def tensor_clamp(self, x: Tensor, num_bits, up=None) -> Tensor:
        if up is not None:
            if up:
                # round up
                round_fn = torch.ceil
            else:
                # round down
                round_fn = torch.floor
        else:
            round_fn = torch.round
        return round_fn(x).clamp(0, 2. ** num_bits - 1.)

    def quantize_to_qtensor_given_scale(self, x:Tensor, scale, zero, num_bits, quantized=True) -> QTensor:
        """Bias Quantization"""
        q_x = x / scale + zero
        q_x = self.tensor_clamp(q_x, num_bits=num_bits)

        return QTensor(q_x, scale=scale, zero=zero, symmetric=False, num_bits=num_bits, quantized=quantized)


class UniformSymmetricQuantization(Quantization):
    """
    Uniform symmetric quantization, used for most weights. Symmetric around 0. zero == 0, always.
    """

    def _quantize_tensor(self, x:Tensor, num_bits, min_val=None, max_val=None) -> Tuple[Tensor, float, float]:
        if not min_val and not max_val:
            min_val, max_val = x.min().item(), x.max().item()

        scale, _ = self.calc_params(min_val=min_val, max_val=max_val, num_bits=num_bits)

        q_x = x / scale

        q_x = self.tensor_clamp(q_x, num_bits=num_bits)

        return q_x, scale, 0

    def quantize_to_qtensor_using_range(self, x: Tensor, num_bits: int, min_val: Optional[float]=None, max_val: Optional[float]=None, quantized: bool=True) -> QTensor:

        q_x, scale, zero = self._quantize_tensor(x, min_val=min_val, max_val=max_val, num_bits=num_bits)
        return QTensor(q_x, scale=scale, zero=zero, symmetric=True, num_bits=num_bits, quantized=quantized)

    def calc_params(self, min_val: float, max_val: float, num_bits: int):
        # Calc Scale
        max_val = max(abs(min_val), abs(max_val))

        qmax = 2. ** (num_bits - 1) - 1.

        scale = max_val / qmax

        return scale, 0

    def tensor_clamp(self, x: Tensor, num_bits: int):
        return x.round().clamp(-(2. ** (num_bits - 1) - 1), 2. ** (num_bits - 1) - 1)

    def quantize_to_qtensor_given_scale(self, x: Tensor, scale: float, zero: int, num_bits: int, quantized: bool=True):
        """Bias Quantization"""

        q_x = x / scale + zero
        q_x = self.tensor_clamp(q_x, num_bits=num_bits)

        return QTensor(q_x, scale=scale, zero=zero, symmetric=True, num_bits=num_bits, quantized=quantized)


class FakeQuant(torch.autograd.Function):
    """
    Simulates quantization error, uses STE for backpropagation.
    For forward pass in quantization aware training: fake-quantized weights and activations
    can then be used normally in dense/conv layer.
    """
    @staticmethod
    def apply_wrapper(*args, handling_qtensors):
        if handling_qtensors:
            num_bits = args[0].num_bits
            args = [args[0]._t] + list(args[1:])

        out, scale, zero = FakeQuant.apply(*args)
        scale, zero = scale.item(), zero.item()

        if handling_qtensors:
            # scale did not actually change, but need to give QTensor these qparams
            # for them to be accessible by NonQuantized layers
            out = QTensor(out, scale, zero, quantized=False, num_bits=num_bits)
        return out

    @staticmethod
    def forward(
            ctx,
            x: Tensor,
            quant: Quantization,
            num_bits: int,
            min_val: float,
            max_val: float,
        ) -> Union[QTensor, Tensor]:
        """
        :param x: torch tensor to quantize
        :param quant: quantization class for tensor
        :param num_bits: number of bits to quantize to
        :param min_val: EMA min_val, for activation quantization with EMA, don't provide for weight quantization
        :param max_val: EMA max_val, for activation quantization with EMA, don't provide for weight quantization
        :return: x: torch tensor
        """
        # NOTE:
        # This forward pass does NOT change the scale or zero point.
        # it only rounds in after affinely transforming to the new scale, new zero
        # but affinely transforms back again (dequantize).
        # The new scale, new zero are given for NonQuantizableModule to access this info
        # (NonQuantizableModule already needs qparams info during QAT)

        if min_val is None or max_val is None:
            min_val, max_val = x.min().item(), x.max().item()

        new_scale, new_zero = quant.calc_params(min_val=min_val, max_val=max_val, num_bits=num_bits)

        # affine transformation and round there to simulate error appropriately
        qx = quant.quantize_to_qtensor_given_scale(
            x, num_bits=num_bits, scale=new_scale, zero=new_zero, quantized=True
        )
        # affinely transform back
        out = qx.dequantize()

        # autograd function may only return tensors, so create one-element tensors for quantization parameters
        return out, Tensor([new_scale]), Tensor([new_zero])

    @staticmethod
    def backward(ctx, grad_output: Tensor, scale: Tensor, zero: Tensor):
        """ Straight Through Estimator """
        return grad_output, None, None, None, None


str2quant = {"uniform": UniformQuantization, "uniform_sym": UniformSymmetricQuantization}


