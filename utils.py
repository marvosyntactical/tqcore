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


