"""
Code by Lisa Kuhn, taken from https://cegit.ziti.uni-heidelberg.de/bklein/dl-frameworks-quantization/

https://arxiv.org/pdf/1712.05877.pdf
https://github.com/google/gemmlowp

Uniform and Symmetric quantizer class to quantize a tensor.

Fake quantization operator for quantization of weights
and activations during forward pass, uses STE for backward pass.

Quantized ReLU for variable zero point.

Also here
https://github.com/eladhoffer/utils.pytorch/blob/master/quantize.py

"""

from collections import namedtuple
from typing import Tuple
import math
import torch
from .qtensor import QTensor

class Quantization():
    def __init__(self, **kwargs):
        pass

    @classmethod
    def from_opt(cls):
        return cls()

    def quantize_to_qtensor(self, x, min_val=None, max_val=None, num_bits=8):
        pass

    def quantize_to_torch_tensor(self, x, min_val=None, max_val=None, num_bits=8):
        pass

    def quantize_to_qtensor_given_scale(self, x:torch.Tensor, scale, zero, num_bits=32) -> QTensor:
        pass

    def calc_params(self, min_val, max_val, num_bits=8):
        pass

    def dequantize_tensor(self, q_x: QTensor):
        pass

    def tensor_clamp(self, x: torch.Tensor, num_bits):
        pass

    def dequantize(self, qt: QTensor):
        pass


class UniformQuantization(Quantization):
    """
    Uniform affine quantization as described in https://arxiv.org/pdf/1712.05877.pdf, section 2.

    Code a la
    https://github.com/google/gemmlowp/blob/master/doc/quantization_example.cc
    """
    def __init__(self, nudge_zero=False):
        self.nudge_zero = nudge_zero

    def _quantize_tensor_using_params(self, x:torch.Tensor, scale, zero=0, num_bits=8):

        q_x = x / scale + zero
        q_x = self.tensor_clamp(q_x, num_bits=num_bits)
        return q_x

    def _quantize_tensor(self, x:torch.Tensor, min_val=None, max_val=None, num_bits=8) -> Tuple[torch.Tensor, float, float]:

        if not min_val and not max_val:
            # weight quantization: dont actually clamp (use full range)
            min_val, max_val = x.min().item(), x.max().item()

        # scale ist real; zero ganz
        scale, zero = self.calc_params(min_val, max_val, num_bits)

        q_x = self._quantize_tensor_using_params(x, scale, zero=zero, num_bits=num_bits)

        return q_x, scale, zero

    def quantize_to_qtensor(self, x, min_val=None, max_val=None, num_bits=8) -> QTensor:

        q_x, scale, zero =  self._quantize_tensor(x, min_val=min_val, max_val=max_val, num_bits=num_bits)
        return QTensor(q_x, scale=scale, zero=zero)

    def quantize_to_qtensor_using_params(self, x, scale, zero=0, num_bits=8) -> QTensor:

        q_x = self._quantize_tensor_using_params(x, scale, zero, num_bits=num_bits)
        return QTensor(q_x, scale=scale, zero=zero)


    def quantize_to_torch_tensor(self, x, qparams, min_val=None, max_val=None, num_bits=8) -> torch.Tensor:
        # updates globally used qparams dict in place instead of returning QTensor

        q_x, scale, zero =  self._quantize_tensor(x, min_val=min_val, max_val=max_val, num_bits=num_bits)

        qparams["scale"] = scale
        qparams["zero"] = zero

        return q_x

    def calc_params(self, min_val, max_val, num_bits=8, nudge=False):
        qmin = 0.
        qmax = 2. ** num_bits - 1.

        max_val = max(max_val, 0.)
        min_val = min(min_val, 0.)

        # scale ist quasi die unit länge in der quantisierten range
        scale = (max_val - min_val) / (qmax - qmin)

        initial_zero = qmin - min_val / scale

        """
        "in both cases (wt, bias), the boundaries [a; b] are nudged so that
        value 0.0 is exactly representable as an integer z(a, b, n)
        after quantization"
        """

        if initial_zero < qmin:
            zero = qmin
        elif initial_zero > qmax:
            zero = qmax
        else:
            if self.nudge_zero or nudge:
                zero = round(initial_zero)

        return scale, zero

    def dequantize_tensor(self, q_x: QTensor) -> torch.Tensor:
        return q_x.scale * (q_x._t.float() - q_x.zero)

    def tensor_clamp(self, x: torch.Tensor, num_bits) -> torch.Tensor:
        return x.round().clamp(0, 2. ** num_bits - 1.)

    def quantize_to_qtensor_given_scale(self, x:torch.Tensor, scale, zero, num_bits=32) -> QTensor:
        """Bias Quantization"""

        q_x = x / scale + zero
        q_x = self.tensor_clamp(q_x, num_bits=num_bits)

        return QTensor(q_x, scale=scale, zero=zero)

class UniformSymmetricQuantization(Quantization):
    """
    Uniform symmetric quantization.
    """

    def _quantize_tensor(self, x:torch.Tensor, min_val=None, max_val=None, num_bits=8) -> Tuple[torch.Tensor, float, float]:
        if not min_val and not max_val:
            min_val, max_val = x.min().item(), x.max().item()

        max_val = max(abs(min_val), abs(max_val))

        scale, _ = self.calc_params(min_val, max_val, num_bits=num_bits)

        q_x = x / scale

        q_x = self.tensor_clamp(q_x, num_bits=num_bits)

        return q_x, scale, 0

    def quantize_to_qtensor(self, x, min_val=None, max_val=None, num_bits=8) -> QTensor:

        q_x, scale, zero = self._quantize_tensor(x, min_val=min_val, max_val=max_val, num_bits=num_bits)
        return QTensor(q_x, scale=scale, zero=zero)

    def calc_params(self, min_val, max_val, num_bits=8):
        # Calc Scale
        max_val = max(abs(min_val), abs(max_val))

        qmax = 2. ** (num_bits - 1) - 1.

        scale = max_val / qmax

        return scale, 0

    def tensor_clamp(self, x: torch.Tensor, num_bits):
        return x.round().clamp(-(2. ** (num_bits - 1) - 1), 2. ** (num_bits - 1) - 1)

    def quantize_to_qtensor_given_scale(self, x, scale, zero=0, num_bits=32):
        """Bias Quantization"""

        q_x = x / scale + zero
        q_x = self.tensor_clamp(q_x, num_bits=num_bits)

        return QTensor(q_x, scale=scale, zero=zero)

    def quantize_to_torch_tensor(self, x, qparams, min_val=None, max_val=None, num_bits=8) -> torch.Tensor:
        # updates globally used qparams dict in place instead of returning QTensor

        q_x, scale, zero =  self._quantize_tensor(x, min_val=min_val, max_val=max_val, num_bits=num_bits)

        qparams["scale"] = scale
        qparams["zero"] = zero

        return q_x

    def dequantize(self, qx: QTensor) -> torch.Tensor:
        return (qx._t - qx.zero) * qx.scale

class FakeQuant(torch.autograd.Function):
    """
    Simulates quantization error, uses STE for backpropagation.
    For forward pass in quantization aware training: fake-quantized weights and activations
    can then be used normally in dense/conv layer.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, quant: Quantization, num_bits: int, min_val: float, max_val: float) -> torch.Tensor:
        """
        :param x: torch tensor to quantize
        :param quant: quantization class for tensor
        :param num_bits: number of bits to quantize to
        :param min_val: EMA min_val, for activation quantization with EMA, don't provide for weight quantization
        :param max_val: EMA max_val, for activation quantization with EMA, don't provide for weight quantization
        :return: x: torch tensor
        """
        x = quant.quantize_to_qtensor(x, num_bits=num_bits, min_val=min_val, max_val=max_val)
        x = quant.dequantize_tensor(x)
        return x # torch.Tensor

    @staticmethod
    def backward(ctx, grad_output):
        """ Straight Through Estimator """
        return grad_output, None, None, None, None


str2quant = {"uniform": UniformQuantization, "uniform_sym": UniformSymmetricQuantization}


