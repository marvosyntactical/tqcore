from enum import Enum
from typing import NamedTuple, Union
import json

"""Options used by various quantization routines, in descending scope"""

class QuantConfigurationError(Exception):
    pass

class TuningMode(Enum):
    # used in ../quantize.py ("which quantization workflow?")
    Calibration = 0
    QAT = 1
    CalibrationThenQAT = 2

class QuantStage(Enum):
    # used in all QuantizableModule s
    # multiple possible flows through this, depending on TuningMode:
    # QAT: 0 -> 2 -> 3
    # Calibration: 0 -> 1 -> 3
    # CalibrationThenQAT: 0 -> 1 -> 2 -> 3
    FP32 = 0
    Calibration = 1
    QAT = 2
    Quantized = 3

class CalibMode(Enum):
    # how to calibrate bounds, if TuningMode is Calibration or CalibrationThenQAT
    # used in QListener
    KL = 0
    EMA = 1

class DistKind(Enum):
    # used by QListener if TuningMode == Calibration and CalibMode == KL
    IGNORE = 0 # for CalibMode == EMA
    CLIPPED = 1 # one sided listener input distribution, e.g. after ReLU6
    SYMM = 2 # symmetric listener input distribution, e.g. after BatchNorm

class ThresholdMode(Enum):
    # used in QListener
    # as in https://arxiv.org/pdf/1906.00532 (there called quantization mode)
    Symmetric = 0
    Independent = 1
    Conjugate = 2

class QConfig(NamedTuple):
    """
    Hyperparameters for QAT and Quantization of
    already trained TSTModel
    """
    # quantization parameters
    num_bits_weight: int = 8
    num_bits: int = 8
    num_bits_bias: int = 32
    leave_first_and_last: bool = False # ignore first and last layer during quantization?
    tuning: str = "qat"
    calib_mode: str = "minandmax"
    thresholds: str = "conjugate"
    calib_num_bins: int = 2048
    calib_eps: int = 5
    record_n_batches_bn: int =  30
    record_n_batches_qlistener: int =  60

    @classmethod
    def from_json(cls, file): # load config from json file
        return cls(**json.load(open(file, "r")))

    @classmethod
    def from_yaml(cls, file): # load config from json file
        return cls(**yaml.load(open(file, "r")))
