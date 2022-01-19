import torch
from torch import Tensor


from .qtensor import QTensor
from .quantization_functions import Quantization, \
        UniformQuantization, UniformSymmetricQuantization, FakeQuant
from .config import QuantStage
from tst.utils import xavier_uniform, _no_grad_uniform_symmetric_

import numpy as np
from typing import Optional, Dict

import os

is_integer = lambda t: ((t.round()==t).all() if t.shape else t.round()==t)


# def one_step_lstm(lstm: torch.nn.Module, data: Tensor) -> Tensor:
#     return out


