import torch
from torch import Tensor


from .qtensor import QTensor
from .quantization_functions import Quantization, \
        UniformQuantization, UniformSymmetricQuantization, FakeQuant
from .config import QuantStage

import numpy as np
from typing import Optional, Dict

import os

is_integer = lambda t: ((t.round()==t).all() if t.shape else t.round()==t)


def one_step_lstm(lstm: torch.nn.Module, data: Tensor) -> Tensor:
    return out

def _no_grad_uniform_symmetric_(tensor, a):
    # avoids using uniform
    with torch.no_grad():
        u_0_1 = torch.rand(
            *tensor.shape,
            dtype=tensor.dtype,
            device=tensor.device,
            requires_grad=tensor.requires_grad
        )
        tensor.data = (u_0_1 - .5) * 2 * a
        return tensor

def xavier_uniform(
        device,
        *shape,
        dtype=torch.float32,
        gain: float = 1.
        ) -> torch.Tensor:
    # creates tensor and fills it with
    # xavier uniformly sampled data
    # (according to
    # https://pytorch.org/docs/stable/nn.init.html?highlight=xavier#torch.nn.init.xavier_uniform_
    # )
    # but uses torch.rand internally because
    # uniform is not supported in ONNX ops
    # https://pytorch.org/docs/stable/onnx.html#supported-operators()
    tensor = torch.zeros(*shape).to(device=device, dtype=dtype)

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation

    return _no_grad_uniform_symmetric_(tensor, a)


