from enum import Enum
from typing import NamedTuple, Union, Dict
import json

"""Options used by various quantization routines, in descending scope"""

class QuantConfigurationError(Exception):
    pass

class QConfig(NamedTuple):
    """
    Hyperparameters for QAT and Quantization of TSTModel
    """
    # quantization parameters
    num_bits: int = 8
    num_bits_weight: int = 8
    num_bits_bias: int = 32

    qadd_rescale: bool = True # whether to rescale after addition overflow in kernel.qadd; overflow is clamped otherwise
    nudge_zero: bool = True # whether to nudge a zero point that is not integer
    leave_first_and_last: bool = False # ignore first and last layer during quantization? # DEPRECATED
    tuning: str = "qat" # qat, calibration, or calibrationthenqat. if using torch.quantization, qat means eager mode, calibration means fx mode
    clamp_ste: bool = False

    calib_mode: str = "minandmax"
    thresholds: str = "conjugate"
    calib_num_bins: int = 2048
    calib_eps: int = 5

    # record only this many QAT batches before freezing these modules:
    record_n_batches_bn: int =  300
    record_n_batches_qlistener: int =  600

    stage_plot_freqs: Dict = {
        "FP32":-1,
        "Calibration":-1,
        "QAT":-1,
        "Quantized":-1,
    }
    stage_plot_indices: Dict = {
        "FP32":[],
        "Calibration":[],
        "QAT":[],
        "Quantized":[],
    }

    transformer: Dict = {} # see example configs/transformer encoder layer
    lstm: Dict = {}

    @classmethod
    def from_json(cls, file):
        cfg = json.load(open(file, "r"))
        return cls(**cfg)


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


