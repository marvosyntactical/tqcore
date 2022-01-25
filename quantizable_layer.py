# Imports
# from torch:
import torch
from torch import Tensor
from torch.fft import fft, fft2
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.utils import _pair

# from builtin modules:
import math
import copy
from enum import Enum
from typing import Optional, Union, Tuple, Dict, Union, Callable, List
from collections import defaultdict
import logging

# for Plotter only:
import os
import logging
import numpy as np
import matplotlib.pyplot as plt

# from tqcore:
from .qtensor import QTensor
from .quantization_functions import Quantization, \
        UniformQuantization, UniformSymmetricQuantization, STE, ClampedSTE
from .config import TuningMode, CalibMode, ThresholdMode, QuantStage, DistKind, QuantConfigurationError
from .histogram import HistogramCalibrator
from .kernel import qadd, qmul

is_integer = lambda t: t.all_close(t.round())

qlogger = logging.getLogger(name="QLayer")

__ASSERT__ = 0

# this module contains quantizable versions of basic nn.Modules, as well as some helper modules
class QuantizableModule(nn.Module):
    """
    Interface for Modules that should be quantized to implement.

    During fp training, this module acts as Identity if forward is not overwritten.
    It also has a Tuning Stage, which can be either
        * Quantization Aware Training (QAT) or
        * Calibration
    and then a fully quantized stage;
    For each of the stages, an Implementation needs to provide the forward method, i.e.:
        1. forward_fp
        2. forward_qat
        3. forward_quantized
    """
    FLOAT_SCALE = 1. # these determine all successive scale / zeros; retrieved by QuantStub
    FLOAT_ZERO = 0.

    # initialize these default values
    # so during very first QAT forward pass, modules can access these attributes.
    scale = FLOAT_SCALE
    zero = FLOAT_ZERO
    # on successive QAT forward passes, QListener has updated these values using recorded stats

    def __init__(self, **qkwargs):
        nn.Module.__init__(self) # super().__init__ somehow calls __init__ of _QBatchNorm for subclasses of it FIXME
        self._set_qkwargs(**qkwargs)

        self.stage_dict = {
            QuantStage.FP32: "FP32",
            QuantStage.Calibration: "Calibration",
            QuantStage.QAT: "QAT",
            QuantStage.Quantized: "Quantized",
        }

        self.n_qat_batches = 0

        self.stage = QuantStage.FP32 # stage starts out at floating point
        self.forward = self.forward_fp # changes from stage to stage

    def _set_qkwargs(
            self,
            quantization: type = UniformQuantization,
            weight_quantization: type = UniformSymmetricQuantization,
            num_bits: int = 8,
            num_bits_weight: int = 8,
            num_bits_bias: int = 32,
            nudge_zero: bool = False,
            clamp_ste: bool = False,
            **qkwargs,
        ):
        self.num_bits = num_bits
        self.quantization = quantization(nudge_zero=nudge_zero)
        self.num_bits_weight = num_bits_weight
        self.weight_quantization = weight_quantization()
        self.num_bits_bias = num_bits_bias
        self.fake_quantize = STE.apply_wrapper if not clamp_ste else ClampedSTE.apply_wrapper

    def stage_str(self) -> str:
        return self.stage_dict[self.stage]

    # OVERWRITE THESE: =======>

    def forward_fp(self, x: Tensor) -> Tensor:
        return x

    def forward_qat(self, x: Optional[QTensor] = None,) -> QTensor:
        self.n_qat_batches += 1
        return x

    def forward_quantized(self, x: QTensor) -> QTensor:
        raise NotImplementedError(f"{type(self)}.forward_quantized")

    def forward_calib(self, *args, **kwargs):
        return self.forward_fp(*args, **kwargs)

    # <======== OVERWRITE THESE

    def calibration_prepare(self):
        self.stage = QuantStage.Calibration
        self.forward = self.forward_calib

    def qat_prepare(self):
        # self._set_qkwargs() # DEBUG: sets defaults

        self.stage = QuantStage.QAT
        self.forward = self.forward_qat

    def quantize(self):
        # input(f"QM: Quantizing quantizable module {self}")
        self.stage = QuantStage.Quantized
        self.forward = self.forward_quantized

        if self.scale==self.FLOAT_SCALE:
            qlogger.warn(f"Quantized {self} has unchanged scale {self.scale}!")

        if not [attr for attr in dir(self) if "scale" in attr or "zero" in attr]:
            qlogger.warn(
f"""
During {self}.quantize(), no scale or zero attribute were found.
These should be set for this instance of {type(self)}
by a QListener
during either entropy calibration (calibration.py) or QAT (qat_*.py).
Could be due to calling super().quantize() before setting self.zero/self.scale.
"""
            )



class QuantStub(QuantizableModule):
    """
    Quantizes incoming torch.Tensors into tqcore.QTensors if necessary.
    Functionality analogous to torch.quantization.QuantStub
    """
    def __init__(self, **qkwargs):
        super().__init__(**qkwargs)

    def forward_qat(self, x):
        super().forward_qat(x)
        if isinstance(x, Tensor)\
                and torch.is_floating_point(x)\
                and not isinstance(x, QTensor):
            r = QTensor(
                x,
                scale=self.scale, # initialized to 1.; update this with a qlistener
                zero=self.zero, # initialized to 0.; update this with a qlistener
                num_bits=self.num_bits,
                quantized=False
            )
        return r

    def forward_quantized(self, x):
        # affine transformation to learned range
        if isinstance(x, Tensor)\
                and torch.is_floating_point(x)\
                and not isinstance(x, QTensor):
            r = self.quantization.quantize_to_qtensor_using_params(
                x=x,
                scale=self.scale,
                zero=self.zero,
                num_bits=self.num_bits,
                quantized=True
            )
        return r

class DeQuantStub(QuantizableModule):
    """
    Dequantizes incoming torch.Tensors into tqcore.QTensors if necessary.
    Functionality analogous to torch.quantization.DeQuantStub
    """
    def __init__(self, **qkwargs):
        super().__init__(**qkwargs)

    def _dequant_outputs(self, outputs, f):
        if not isinstance(outputs, tuple):
            assert isinstance(outputs, QTensor), type(outputs)
            outputs = (outputs,)

        outputs = list(outputs)

        for i, out in enumerate(outputs):
            if isinstance(out, QTensor):
                assert out.num_bits==self.num_bits, (out.num_bits, self.num_bits)
                outputs[i] = f(out)

        if len(outputs) == 1:
            outputs = outputs[0]
        else:
            outputs = tuple(outputs)
        return outputs

    def forward_qat(self, outputs: Union[QTensor, Tuple[QTensor]]):
        super().forward_qat()
        f = lambda qt: qt._t
        return self._dequant_outputs(outputs, f)

    def forward_quantized(self, outputs: Union[QTensor, Tuple[QTensor]]):
        f = lambda qt: qt.dequantize()
        return self._dequant_outputs(outputs, f)

class QMul(QuantizableModule):
    def __init__(self, type_a="activ", type_b="activ", **qkwargs):
        super().__init__(**qkwargs)
        for c, typ in zip("ab", [type_a, type_b]):
            typ = str(typ).lower()
            msg = f"Specify if tensor is of type activation, weight or bias (Got type_{c}={typ})"
            if "a" in typ:
                quant = self.quantization
                nb = self.num_bits
            elif "w" in typ:
                quant = self.weight_quantization
                nb = self.num_bits_weight
            else:
                assert "b" in typ, msg
                quant = self.weight_quantization
                nb = self.num_bits_bias

            setattr(self, "quant_"+c, quant)
            setattr(self, "num_bits_"+c, nb)

    def forward_fp(self, a: torch.Tensor, b: torch.Tensor):
        return a * b

    def forward_qat(self, a: QTensor, b: QTensor):
        # NO affine transformation; multiply in float but keep track of scale
        super().forward_qat()
        assert a.num_bits==self.num_bits_a, (a.num_bits, self.num_bits_a)
        assert b.num_bits==self.num_bits_b, (b.num_bits, self.num_bits_b)
        rfloat = a._t * b._t
        r = QTensor(rfloat, scale=self.scale, zero=self.zero, num_bits=self.num_bits, quantized=False)
        return r

    def forward_quantized(
            self,
            a: Union[QTensor, torch.Tensor, nn.Parameter, float, int],
            b: Union[QTensor, torch.Tensor, nn.Parameter, float, int],
        ):
        # affine transformation; simulates low bit multiplication
        return qmul(
            a=a, b=b, factor=1.,
            scale_next=self.scale, zero_next=self.zero, op=torch.mul,
            quant_a=self.quant_a, quant_b=self.quant_b,
            num_bits_a=self.num_bits_a, num_bits_b=self.num_bits_b
        )


class QMatMul(QuantizableModule):
    def __init__(self, type_a="activ", type_b="activ", factor: float=1., **qkwargs):
        super().__init__(**qkwargs)
        self.factor = float(factor)
        for c, typ in zip("ab", [type_a, type_b]):
            typ = str(typ).lower()
            msg = "Specify if tensor is of type activation, weight or bias"
            if "a" in typ:
                quant = self.quantization
                nb = self.num_bits
            elif "w" in typ:
                quant = self.weight_quantization
                nb = self.num_bits_weight
            else:
                assert "b" in typ, msg
                quant = self.weight_quantization
                nb = self.num_bits_bias

            setattr(self, "quant_"+c, quant)
            setattr(self, "num_bits_"+c, nb)

    def forward_fp(self, a, b):
        return self.factor * torch.matmul( a , b )

    def forward_qat(self, a, b):
        super().forward_qat()
        assert a.num_bits==self.num_bits_a, (a.num_bits, self.num_bits_a)
        assert b.num_bits==self.num_bits_b, (b.num_bits, self.num_bits_b)
        r = self.factor * torch.matmul(a._t, b._t)
        r = QTensor(r, scale=self.scale, zero=self.zero, num_bits=self.num_bits, quantized=False)
        return r

    def forward_quantized(self, a: QTensor, b: QTensor) -> QTensor:
        return qmul(
            a=a, b=b, factor=self.factor,
            scale_next=self.scale, zero_next=self.zero, op=torch.matmul,
            quant_a=self.quant_a, quant_b=self.quant_b,
            quant_out=self.quantization,
            num_bits_a=self.num_bits_a, num_bits_b=self.num_bits_b,
            num_bits_out=self.num_bits
        )

class QLinear(QuantizableModule, nn.Linear):
    """
    Quantizable Version of torch.nn.Linear
    """
    def __init__(self, *args, qkwargs: Dict, dont_fakeQ: bool=False, **kwargs):
        # super().__init__(**qkwargs)
        QuantizableModule.__init__(self, **qkwargs)
        nn.Linear.__init__(self, *args, **kwargs)

        self.dont_fakeQ = dont_fakeQ

    def forward_fp(self, x: Tensor):
        return F.linear(x, self.weight, self.bias)

    def forward_qat(self, x: QTensor):
        assert x.num_bits==self.num_bits, (x.num_bits, self.num_bits)
        if not self.dont_fakeQ:
            # fake quantize w&b using min, max of the parameters as range
            self.weight.data = self.fake_quantize(
                self.weight.data, self.weight_quantization, self.num_bits_weight, None, None, handling_qtensors=False
            )

            if self.bias is not None:
                self.bias.data = self.fake_quantize(
                    self.bias.data, self.weight_quantization, self.num_bits_bias, None, None, handling_qtensors=False
                )
        return QTensor(F.linear(x._t, self.weight, self.bias), scale=self.scale, zero=self.zero, num_bits=self.num_bits, quantized=False)

    def quantize(self):
        super().quantize()
        # nn.Parameter weight is replaced by QTensor
        self.w = self.weight_quantization.quantize_to_qtensor_using_range(
            x=self.weight.data,
            num_bits=self.num_bits_weight,
            quantized=True
        )

    def forward_quantized(self, x:QTensor):
        assert x.num_bits==self.num_bits, (x.num_bits, self.num_bits)

        w = self.w

        if self.bias is not None:
            b = self.weight_quantization.quantize_to_qtensor_given_scale(
                x=self.bias,
                scale=w.scale * x.scale,
                zero=0,
                num_bits=self.num_bits_bias,
                quantized=True
            )
            b = b._t
        else:
            b = None

        x_zeroed = x._t - x.zero
        w_zeroed = w._t - w.zero

        # low bitwidth w @ x + b forward pass
        out = F.linear(x_zeroed, w_zeroed, bias=b)

        # rescale (see jacob et al 2018, sec 2.2)
        multiplier = (x.scale * w.scale) / self.scale

        # scale result tensor back to given bit width
        out = out * multiplier + self.zero

        # round and clamp values
        out = self.quantization.tensor_clamp(x=out, num_bits=self.num_bits)

        return QTensor(out, scale=self.scale, zero=self.zero, num_bits=self.num_bits, quantized=True)


class QAdd(QuantizableModule):
    def __init__(self, type_a="activ", type_b="activ", **qkwargs):
        self.rescale = qkwargs["qadd_rescale"]
        super().__init__(**qkwargs)
        for c, typ in zip("ab", [type_a, type_b]):
            typ = str(typ).lower()
            msg = "Specify if tensor is of type activation, weight or bias"
            if "a" in typ:
                quant = self.quantization
                nb = self.num_bits
            elif "w" in typ:
                quant = self.weight_quantization
                nb = self.num_bits_weight
            else:
                assert "b" in typ, msg
                quant = self.weight_quantization
                nb = self.num_bits_bias

            setattr(self, "quant_"+c, quant)
            setattr(self, "num_bits_"+c, nb)

    def forward_fp(self, a, b):
        return a + b

    def forward_qat(self, a: QTensor, b: QTensor):
        assert a.num_bits==self.num_bits_a, (a.num_bits, self.num_bits_a)
        assert b.num_bits==self.num_bits_b, (b.num_bits, self.num_bits_b)
        super().forward_qat()
        rfloat = a._t + b._t
        r = QTensor(rfloat, scale=self.scale, zero=self.zero, num_bits=self.num_bits, quantized=False)
        return r

    def forward_quantized(self, a: QTensor, b:QTensor) -> QTensor:
        return qadd(
            a=a, b=b,
            scale_next=self.scale,
            zero_next=self.zero,
            op=torch.add,
            quant_a=self.quant_a,
            quant_b=self.quant_b,
            quant_out=self.quantization,
            num_bits_a=self.num_bits_a,
            num_bits_b=self.num_bits_b,
            num_bits_out=self.num_bits,
            rescale=self.rescale
        )

class QStack(QuantizableModule):
    """
    Quantizable version of torch.stack; for LSTM..
    May be used with the kwargs "dim" and "out" of torch.stack.
    """
    def forward_quantized(self, qtensors: List[QTensor], dim: int=0, out: Tensor=None) -> QTensor:
        requantized: List[Tensor] = []
        for qt in qtensors:
            assert type(qt) == QTensor, type(qt)
            assert qt.num_bits == self.num_bits, (qt.num_bits, self.num_bits)
            fp_tensor = qt.dequantize()
            rq = fp_tensor / self.scale + self.zero
            rq = self.quantization.tensor_clamp(x=rq, num_bits=self.num_bits)
            requantized += [rq]
        r: Tensor = torch.stack(requantized, dim=dim, out=out)
        qr: QTensor = QTensor(r, scale=self.scale, zero=self.zero, quantized=True, num_bits=self.num_bits)
        return qr

    def forward_fp(self, *args, **kwargs):
        return torch.stack(*args, **kwargs)

    def forward_qat(self, qtensors: List[QTensor], dim: int=0, out: Tensor=None):
        super().forward_qat()
        for qt in qtensors:
            assert type(qt) == QTensor, type(qt)
            assert qt.num_bits == self.num_bits, (qt.num_bits, self.num_bits)
        r = torch.stack([qt._t for qt in qtensors], dim=dim, out=out)
        return QTensor(r, scale=self.scale, zero=self.zero, quantized=False, num_bits=self.num_bits)


class QCat(QuantizableModule):
    """
    Quantizable version of torch.cat;
    May be used with the kwargs "dim" and "out" of torch.cat.
    """
    def forward_fp(self, *args, **kwargs):
        return torch.cat(*args, **kwargs)

    def forward_qat(self, qtensors: List[QTensor], dim: int=0, out: Tensor=None):
        super().forward_qat()
        for qt in qtensors:
            assert type(qt) == QTensor, type(qt)
            assert qt.num_bits == self.num_bits, (qt.num_bits, self.num_bits)
        r = torch.cat([qt._t for qt in qtensors], dim=dim, out=out)
        return QTensor(r, scale=self.scale, zero=self.zero, quantized=False, num_bits=self.num_bits)

    def forward_quantized(self, qtensors: List[QTensor], dim: int=0, out: Tensor=None) -> QTensor:
        requantized: List[Tensor] = []
        for qt in qtensors:
            assert type(qt) == QTensor, type(qt)
            assert qt.num_bits == self.num_bits, (qt.num_bits, self.num_bits)
            fp_tensor = qt.dequantize()
            rq = fp_tensor / self.scale + self.zero
            rq = self.quantization.tensor_clamp(x=rq, num_bits=self.num_bits)
            requantized += [rq]
        r: Tensor = torch.cat(requantized, dim=dim, out=out)
        qr: QTensor = QTensor(r, scale=self.scale, zero=self.zero, quantized=True, num_bits=self.num_bits)
        return qr





class QFill(QuantizableModule):
    """
    Quantizable version of torch.masked_fill.
    fp_neg_val: negative value to fill with in floating point instead of negative infinity; to keep EMA range bounded
    """
    def __init__(self, fp_neg_val: float=-1e5, **qkwargs):
        super().__init__(**qkwargs)
        self.fp_neg_val = float(fp_neg_val)

    def forward_fp(self, scores, mask):
        scores = scores.masked_fill(mask==torch.as_tensor(False), self.fp_neg_val)
        return scores

    def forward_qat(self, scores, mask):
        super().forward_qat()
        assert scores.num_bits == self.num_bits, (scores.num_bits, self.num_bits)
        scores = scores._t.masked_fill(mask==torch.as_tensor(False), self.fp_neg_val)
        return QTensor(scores, scale=self.scale, zero=self.zero, quantized=False, num_bits=self.num_bits)

    def forward_quantized(self, scores, mask):
        assert scores.num_bits == self.num_bits, (scores.num_bits, self.num_bits)
        scores = scores.masked_fill(mask==torch.as_tensor(False), self.zero)
        return QTensor(scores, scale=self.scale, zero=self.zero, num_bits=self.num_bits, quantized=True)

class QBoolMask(QuantizableModule):
    # NOTE does not require a listener/qparam attributes.
    # TODO: either
    # 1. distinguish between QuantizableModules which need a QListener or not
    # 2. implement this module as a method of QTensor instead
    def forward_fp(self, x: Tensor, mask: torch.BoolTensor):
        return x * mask

    def forward_qat(self, x: QTensor, mask: torch.BoolTensor):
        super().forward_qat()
        assert x.num_bits == self.num_bits, (x.num_bits, self.num_bits)
        r = x._t * mask
        return QTensor(r, scale=x.scale, zero=x.zero, num_bits=self.num_bits, quantized=False)

    def forward_quantized(self, x: QTensor, mask: torch.BoolTensor):
        assert x.num_bits == self.num_bits, (x.num_bits, self.num_bits)
        r = x._t.masked_fill(mask==torch.as_tensor(False), x.zero)
        return QTensor(r, scale=x.scale, zero=x.zero, num_bits=self.num_bits, quantized=True)

class QSoftmax(QuantizableModule):
    def __init__(self, dim: int=-1, time_window: int = 144, alpha: float = None, layer_num: int=0, **qkwargs):
        raise DeprecationWarning("QSoftmax is deprecated, use NonQuantizableModuleWrap to wrap a normal softmax instead")

        super().__init__(**qkwargs)
        self.dim = int(dim)

        if alpha is None:
            # take maximal alpha such that summed denominator of int8s is representable in int32
            # (want alpha to be large to have low temp)
            # alpha = (torch.log(torch.Tensor([2147483647/time_window]))/((2**num_bits)-1)).item()
            # alpha = 0.02
            alpha = 0.02173043

        self.alpha = float(abs(alpha))

        # this records EMA stats of exponential, and is not used for fake quant effect
        self.exp_listener = QListener(
            self,
            listener_name="exp",
            function=self._scaled_exp,
            plot_name="softmax exp"+ str(layer_num),
            dont_fakeQ=False,
            **qkwargs
        )

        # this records stats of normed output (which will be between 0 and 1 anyway),
        # but is used only for fake quantizing
        self.norm_listener = QListener(self, plot_name="softmax normed"+ str(layer_num), **qkwargs)

    def _scaled_exp(self, inp: Union[Tensor, QTensor]) -> Union[Tensor, QTensor]:
        # (differentiable)
        if isinstance(inp, Tensor):
            exponent = inp * self.alpha
        else:
            assert inp.num_bits==self.num_bits
            exponent = inp._t * self.alpha
        # out = torch.exp(exponent.clamp(max=78.))
        out = torch.exp(exponent)

        assert not torch.isinf(out).sum(), f"QSoftmax._scaled_exp produced a ratio of {torch.isinf(out).sum()/out.nelement()} infs"
        return out

    def _set_exp_lkp(self):
        # this range is tailored to UniformQuantization
        self.EXP_STEP_FUN = self._scaled_exp(torch.arange(2.**self.num_bits)).round()

        # debug:
        # import matplotlib.pyplot as plt
        # plt.plot(self.EXP_STEP_FUN.detach().numpy())
        # plt.show()

    def forward_fp(self, inp: torch.Tensor) -> torch.Tensor:
        out = inp.softmax(dim=self.dim)
        return out

    def forward_qat(self, inp: QTensor) -> torch.Tensor:
        super().forward_qat()
        assert inp.num_bits == self.num_bits, (inp.num_bits, self.num_bits)
        self.exp_listener(inp)
        # LogSumExp trick in comment
        numerator = self._scaled_exp(inp._t) # -inp._t.max())
        out = numerator/numerator.sum(dim=self.dim).unsqueeze(self.dim)

        # scale and zero are arbitrary here
        out = QTensor(out, scale=1., zero=0., quantized=False)
        # fake quantize:
        out = self.norm_listener(out)
        # # self.scale, self.zero are set by norm_listener
        # out = QTensor(out, scale=self.scale, zero=self.zero, quantized=False)
        return out

    def quantize(self):
        super().quantize()
        self._set_exp_lkp()

    def _exp_lkp(self, long_tensor: torch.LongTensor):
        # (not differentiable)
        assert hasattr(self, "EXP_STEP_FUN"), f"{self}._set_exp_lkp() must be called before it can perform a lookup"
        return QTensor(
            self.EXP_STEP_FUN[long_tensor], scale=self.exp_scale, zero=self.exp_zero,
            quantized=True
        )

    def forward_quantized(self, inp: QTensor) -> QTensor:
        assert inp.num_bits==self.num_bits, (inp.num_bits, self.num_bits)

        exponentiated: QTensor = self._exp_lkp(inp._t.long()) # NOTE: range from 0-255 here already!
        zeroed_exp = exponentiated._t - exponentiated.zero
        # M = exponentiated.scale / self.normed_scale
        norm_scale = 1./(2**self.num_bits)
        M = exponentiated.scale / norm_scale
        normed_exp = (zeroed_exp * M) # + self.normed_zero
        r = self.quantization.tensor_clamp(x=normed_exp, num_bits=self.num_bits)
        return r


class NonQuantizableModuleWrap(QuantizableModule):
    """
    NonQuantizableModuleWrap can be used to wrap a module that should stay in floating point after
    the model is converted to the quantized stage. It does not need a QListener.
    Example usage:
        >>> self.qsoftmax = NonQuantizableModuleWrap(
        >>>     nn.Softmax(dim=-1), **qkwargs
        >>> )

    """

    def __init__(self, module, plot_name=None, **qkwargs):
        super().__init__(**qkwargs)

        self.fp_module = module

        self.out_listener = QListener(
            self,
            dont_fakeQ=False,
            plot_name=plot_name,
            clipped_distr=False, # NOTE FIXME this is specifically for MHATTN; make custom for non quant relu etc
            **qkwargs
        )

    def quantize(self):
        super().quantize()
        assert self.stage == QuantStage.Quantized

    def forward_fp(self, *args, **kwargs) -> torch.Tensor:
        r = self.fp_module(*args, **kwargs)
        return r

    def forward_calib(self, x, *args, **kwargs):
        r = self.fp_module(x, *args, **kwargs)
        return self.out_listener(r)

    def forward_qat(self, *args, **kwargs) -> torch.Tensor:
        super().forward_qat()

        for arg in args:
            assert not isinstance(arg, Tensor)
            if isinstance(arg, QTensor):
                assert arg.num_bits==self.num_bits, (arg.num_bits, self.num_bits)

        fp_args = [a._t if isinstance(a, QTensor) else a for a in args]

        # forward in FP32:
        fp_out = self.fp_module(*fp_args, **kwargs)

        if not isinstance(fp_out, tuple):
            fp_out = (fp_out,)

        fakeq_out = [
                    self.out_listener(
                        QTensor(fp_o, scale=self.scale, zero=self.zero, num_bits=self.num_bits, quantized=False)
                    )
                if not isinstance(fp_o, (QTensor, type(None)))
                else fp_o
            for fp_o in fp_out
        ]

        return tuple(fakeq_out) if len(fakeq_out) > 1 else fakeq_out[0]

    def forward_quantized(self, *args, **kwargs) -> QTensor:
        for arg in args:
            assert not isinstance(arg, Tensor)
            if isinstance(arg, QTensor):
                assert arg.num_bits==self.num_bits, (arg.num_bits, self.num_bits)

        fp_args = [a.dequantize() if isinstance(a, QTensor) else a for a in args]

        fp_outs = self.fp_module(*fp_args, **kwargs)
        if not isinstance(fp_outs, tuple):
            fp_outs = (fp_outs,)

        q_outs = [
            self.quantization.quantize_to_qtensor_using_params(
                x=fp_out,
                scale=self.scale,
                zero=self.zero,
                num_bits=self.num_bits,
                quantized=True
            )
            if not isinstance(fp_out, QTensor)
            else fp_out for fp_out in fp_outs
        ]

        return tuple(q_outs) if len(q_outs) > 1 else q_outs[0]


class QReLU6(QuantizableModule):
    """
    Quantizable version of ReLU6.
    To be used instead of ReLU to provide natural range during FP32 already
    According to tensorflow guide: TODO
    """
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

    def forward_fp(self, x: torch.Tensor) -> torch.Tensor:
        out = nn.functional.relu6(x)
        return out

    def forward_qat(self, x: QTensor) -> QTensor:
        super().forward_qat()
        assert x.num_bits==self.num_bits, (x.num_bits, self.num_bits)
        out = nn.functional.relu6(x._t)
        out =  QTensor(out, scale=x.scale, zero=x.zero, num_bits=self.num_bits, quantized=False)
        return out

    def forward_quantized(self, x: QTensor) -> QTensor:
        assert x.num_bits==self.num_bits, (x.num_bits, self.num_bits)

        scale = self.scale
        zero = self.zero

        # this should be the case if self's QListener records the previous module
        # as well. the below rescaling is therefore usually unnecessary.
        # In the case that these lines raise an AssertionError,
        # either:
        # 1. make QListener listen to previous module as well
        # (init w/ e.g. QListener(linear, relu, **qkwargs))
        # 2. comment out these assertions
        assert round(scale, 5) == round(x.scale, 5), \
                (scale, x.scale)
        assert round(zero, 5) == round(x.zero, 5), \
                (zero, x.zero)

        # requantize
        inp = x.dequantize()
        inp = inp / scale + zero
        # clamp
        six = round(6. / scale + zero)
        out = inp.round().clamp(min=zero, max=six)

        out =  QTensor(out, scale=scale, zero=zero, num_bits=self.num_bits, quantized=True)
        return out


class QHardSigmoid(QuantizableModule):

    def forward_fp(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.hardsigmoid(x)

    def forward_qat(self, x: QTensor) -> QTensor:
        super().forward_qat()
        assert x.num_bits==self.num_bits, (x.num_bits, self.num_bits)
        r = nn.functional.hardsigmoid(x._t)
        return QTensor(r, scale=self.scale, zero=self.zero, num_bits=self.num_bits, quantized=False)

    def forward_quantized(self, x: QTensor) -> QTensor:
        assert x.num_bits==self.num_bits, (x.num_bits, self.num_bits)

        scale = self.scale
        zero = self.zero

        # omit zero here and add it at the end
        one = round(1. / scale )
        minus_three = round(-3. / scale )
        plus_three = round(3. / scale )
        a_half = round(.5 / scale)

        out = x.dequantize()
        out = out / scale # + zero

        index_smaller = out <= minus_three + zero
        index_larger = out >= plus_three + zero
        idx_middle = ~(index_smaller | index_larger)

        out[index_smaller] = zero
        out[index_larger] = one + zero
        out[idx_middle] *= 1/.6
        out[idx_middle] += a_half + zero

        # TODO plot and see if ceil or floor is better!
        out.round_()

        return QTensor(out, scale=scale, zero=zero, quantized=True)

class QHardTanH(QuantizableModule):

    def forward_fp(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.hardtanh(x)

    def forward_qat(self, x: QTensor) -> QTensor:
        assert x.num_bits==self.num_bits, (x.num_bits, self.num_bits)
        super().forward_qat()
        r = nn.functional.hardtanh(x._t)
        return QTensor(r, scale=self.scale, zero=self.zero, quantized=False)

    def forward_quantized(self, x: QTensor) -> QTensor:
        assert x.num_bits==self.num_bits, (x.num_bits, self.num_bits)

        scale = self.scale
        zero = self.zero

        # omit zeros here and add it at the end
        minus_one = round(-1. / scale ) + zero
        plus_one = round(1. / scale ) + zero

        out = x.dequantize()
        out = out / scale + zero

        index_smaller = out <= minus_one
        index_larger = out >= plus_one
        idx_middle = ~(index_smaller | index_larger)

        out[index_smaller] = minus_one
        out[index_larger] = plus_one

        out.round_()

        return QTensor(out, scale=scale, zero=zero, quantized=True)

def DiscreteHartleyTransform(input):
    # from https://github.com/AlbertZhangHIT/Hartley-spectral-pooling/blob/master/spectralpool.py
    # TODO test as alternative to FFT
    fft = torch.rfft(input, 2, normalized=True, onesided=False)
    dht = fft[:, :, :, :, -2] - fft[:, :, :, :, -1]
    return dht

class FFT(QuantizableModule):
    # TODO figure out mask

    # figure out whether to use kwarg "normalize" (scale by 1/sqrt(n))
    # paper: vandermonde matrix has normalization
    # third party code: no normalization

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward_fp(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if mask is not None:
            qlogger.warn(f"FFT mask not yet implemented!")

        # method 1 (paper):
        x = fft(fft(x, dim=-1), dim=-2).real
        # method 2:
        # x = fft2(x)
        # x = x.real # method 2b:  + x.imag
        return x


class QLabel(QuantizableModule):
    """
    Learnable Label (https://arxiv.org/abs/2109.10252)
    appended/prepended to input and fitted to its range
    """
    def __init__(self,
            rank: int = 3,
            dim: int = 1,
            hidden_dim: int = 100,
            prepend: bool = False,
            init_fn: Callable = torch.randn,
            plot_name: Optional[str] = None,
            **qkwargs
        ):
        """
        :param rank: rank of tensor x to concatenate the label to
        :param cat_dim: which dimension of x to concatenate the label to
        :param hidden_dim: which dimension of x is the hidden dimension
        :param hidden_size: hidden dimensionality of label; must match x.shape[hidden_dim]
        :param prepend: whether to prepend or append the label in given dim
        :param init_fn:
        """
        super().__init__(**qkwargs)

        self.rank = rank
        self.cat_dim = cat_dim
        self.hidden_dim = hidden_dim

        dims = [1] * rank
        dims[hidden_dim]=hidden_size

        self.prepend = prepend

        self.label = nn.Parameter(init_fn(*dims))
        self.qcat = QCat(**qkwargs)

        self.qcat_listener = QListener(self.qcat, plot_name=plot_name, **qkwargs)

        self.num_bits_label = self.num_bits_weight

    def _fwd(self, X: torch.Tensor):

        expand_args = [-1] * self.rank
        # expand along all dimensions which arent hidden_dim or cat_dim
        for d in range(self.rank):
            if d not in {self.hidden_dim, self.cat_dim}:
                expand_args[d] = X.shape[d]

        expanded_label = self.label.expand(*expand_args)

        if self.prepend:
            to_cat = [expanded_label, X]
        else:
            to_cat = [X, expanded_label]

        # Add label
        x_with_label = self.qcat(to_cat, dim=self.cat_dim)
        return x_with_label

    def forward_fp(self, X: Tensor):
        return self._fwd(X)

    def forward_qat(self, X: QTensor):
        super().forward_qat()
        assert X.num_bits == self.num_bits, (X.num_bits, self.num_bits)
        self.fake_quantize(
            self.label.data, self.weight_quantization, self.num_bits_label, None, None, handling_qtensors=False
        )
        x_with_label = self._fwd(x)
        x_with_label = self.qcat_listener(x_with_label)
        return x_with_label

    def forward_quantized(self, X: QTensor):
        return self._fwd(X)

    def quantize(self):
        super().quantize()
        # nn.Parameter label is replaced by QTensor
        self.label = self.weight_quantization.quantize_to_qtensor_using_range(
            x=self.label.data,
            num_bits=self.num_bits_label, # _bias # TODO
            quantized=True
        )


class QPositionalEncoding(QuantizableModule):
    """
    Learnable Position Encoding (A W x D bias matrix that we add to the input)
    """
    def __init__(self,
            dim: int = 0,
            time_window: int = 24,
            init_fn: Callable = torch.rand,
            **qkwargs
        ):
        super().__init__(**qkwargs)

        self.rescale = qkwargs["qadd_rescale"]
        self.W = nn.Parameter(init_fn(time_window, dim))

    def forward_fp(self, X: torch.Tensor):
        """
        Encode inputs.
        Args:
            X (FloatTensor): Sequence of word vectors
                ``(batch_size, dim, time_window)``
        """
        # Add position encodings
        return self.W + X

    def forward_qat(self, X: QTensor):
        super().forward_qat()
        assert X.num_bits == self.num_bits, (X.num_bits, self.num_bits)
        self.fake_quantize(
            self.W.data, self.weight_quantization, self.num_bits_bias, None, None, handling_qtensors=False
        )
        rfloat = self.W + X._t
        r = QTensor(rfloat, scale=self.scale, zero=self.zero, num_bits=self.num_bits, quantized=False)
        return r

    def forward_quantized(self, X: QTensor):
        return qadd(
            a=X, b=self._w,
            scale_next=self.scale, zero_next=self.zero, op=torch.add,
            quant_a=self.quantization, quant_b=self.weight_quantization, # use uniform quantization for bias weight
            quant_out=self.quantization,
            num_bits_a=self.num_bits,
            num_bits_b=self.num_bits_bias, # _bias, # TODO
            num_bits_out=self.num_bits,
            rescale=self.rescale
        )

    def quantize(self):
        super().quantize()
        # nn.Parameter W is replaced by QTensor
        self._w = self.weight_quantization.quantize_to_qtensor_using_range(
            x=self.W.data,
            num_bits=self.num_bits_bias, # _bias # TODO
            quantized=True
        )

class QListener(QuantizableModule):
    """
    During tuning (Calibration or QAT),
    this module records either of (min and max) or (mean) or (a histogram)
    of torch.Tensors passing through.
    (no other module records stats)
    Constructed with n nn.Modules "modules", for each of which the QListener sets the attributes:
        - module.scale
        - module.zero
    and either
        - module.min
        - module.max
    (if self.calibration_mode == CalibMode.EMA)
    or
        - module.mu
    (if self.distribution_kind == DistKind.SYMM)
    or
        - Nothing else
    (if self.distribution_kind == DistKind.CLIPPED)
    If name is given, this module sets the attributes module.{self.name}_scale, and so on, instead.
    Each QuantizableModule needs to be updated by some QListener, that should manually be placed
    after the module.
    """
    def __init__(
            self,
            *modules: nn.Module,
            listener_name = None,
            function = None,
            dont_fakeQ: bool = False,
            ema_decay: float = .9999,
            nudge_zero: bool = False,
            calibration: Optional[str] = None, # manually override qkwargs["calib_mode"]
            clipped_distr: Optional[bool] = None, # manually override distribution type
            record_n_batches: int = 999999999,
            plot_name: Optional[str] = None,
            **qkwargs
        ):

        super().__init__(**qkwargs)

        # print(f"qlistener {plot_name} setup: qkwargs={qkwargs}\nself.num_bits={self.num_bits}\nself.num_bits_weight={self.num_bits_weight}")

        self.function = function # optionally apply function before collecting stats (for softmax)
        self.name = "" if listener_name is None else str(listener_name) # set attribute name_scale_next and so on
        self.ema_decay = ema_decay
        self.set_qparams_during_quantization = False # updates qparams after every tuning forward call if False; otherwise only updates qparams Once during conversion
        self.mods = list(modules)
        self.record_n_batches = qkwargs["record_n_batches_qlistener"]

        # attach plotter to self if plot_name is given
        if plot_name is not None:
            self.qplotter = QPlotter(plot_name=plot_name, **qkwargs)
            self.mods += [self.qplotter] # set stats
        else:
            self.qplotter = lambda x, **kwargs: x

        # CalibMode
        calib_mode = qkwargs["calib_mode"].lower() if calibration is None else calibration
        if "kl" in calib_mode or "div" in calib_mode or "ent" in calib_mode:
            self.calibration_mode = CalibMode.KL
        elif "min" in calib_mode or "max" in calib_mode or "ema" in calib_mode:
            self.calibration_mode = CalibMode.EMA
        else:
            raise QuantConfigurationError(f"calib_mode={calib_mode}")

        if self.calibration_mode==CalibMode.KL:
            self.hist_calib = HistogramCalibrator(
                num_bits=self.num_bits,
                num_bins=qkwargs["calib_num_bins"],
                unsigned=True,
                skip_zeros=False,
            )
            # type of input distribution
            if clipped_distr is not None:
                # manually specified by user during init
                self.distribution_kind = DistKind.CLIPPED if clipped_distr else DistKind.SYMM
            else:
                # infer distribution based on input modules
                errmsg = f"""QListener cannot infer kind of input distribution on its own :/
It listens to {self.mods}.
Please specify the kind of input distribution for this QListener
by initializing it with clipped_distr: bool given as kwarg.
"""
                # is there a clipping input module?
                clipping = True in [isinstance(mod, tuple(CLIPPING_MODULES)) for mod in self.mods]
                # is there a symmetrizing input module?
                symm = True in [isinstance(mod, tuple(SYMMETRIZING_MODULES)) for mod in self.mods]

                if clipping and not symm:
                    self.distribution_kind = DistKind.CLIPPED
                    distkind = "clipped"
                elif symm and not clipping:
                    self.distribution_kind = DistKind.SYMM
                    distkind = "symmetric"
                else:
                    if not clipping and not symm:
                        errmsg += "(The problem could be due to the modules above being unknown.)"
                        raise QuantConfigurationError(errmsg)
                    else:
                        # errmsg += "(The problem is due to some input modules producing a symmetric, others a clipped distribution.)"
                        # NOTE assume clipping is applied to everything; this occurs in a qlistener that listens to a relu after a batchnorm
                        qlogger.warn(f"QListener encountered some clipping, some symmetrizing input modules. Assuming everything is clipped.")
                        self.distribution_kind = DistKind.CLIPPED
                        distkind = "clipped"
                qlogger.warn(f"QListener listening to\n{self.mods}\ndecided on {distkind}")
        else:
            # distribution does not matter if we just estimate EMA min/max
            if clipped_distr is not None:
                qlogger.warn(f"QListener listening to\n{self.mods}\nignores given kwarg clipped_distr={clipped_distr} because it calibrates with EMA only")
            self.distribution_kind = DistKind.IGNORE

        if self.distribution_kind != DistKind.CLIPPED:
            self.__stats__ = defaultdict()

        threshs = qkwargs["thresholds"].lower()
        if "symm" in threshs:
            self.threshold_mode = ThresholdMode.Symmetric
        elif "indep" in threshs:
            self.threshold_mode = ThresholdMode.Independent
        elif "conj" in threshs:
            self.threshold_mode = ThresholdMode.Conjugate
        else:
            raise QuantConfigurationError(f"thresholds={threshs}")

        self.dont_fakeQ = dont_fakeQ

        self._qparams_set = False
        # print(f"qlistener {plot_name} setup ended;\nself.num_bits={self.num_bits}\nself.num_bits_weight={self.num_bits_weight}")

    def freeze(self):
        qlogger.info(f"{self} stopped recording.")
        assert not self.set_qparams_during_quantization, "stats must have been set during qat to freeze"

        # TODO call super().forward_qat .. how can I use self in method?
        self.forward_qat = lambda x: x

    def forward_fp(self, x: Tensor, dont_plot=False):
        x = self.qplotter(x, dont_plot=dont_plot)
        return x

    def forward_calib(self, x: Tensor, dont_plot=False):
        """
        Save histograms for Calibration as in
        https://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf
        """
        assert self.stage == QuantStage.Calibration
        if self.calibration_mode == CalibMode.KL:
            self.hist_calib.collect(x)
        if not self.distribution_kind == DistKind.CLIPPED:
            # update ema for mean or min+max
            self._update_ema(x)
        if not self.set_qparams_during_quantization:
            self._set_qparams()
        x = self.qplotter(x, dont_plot=dont_plot)
        return x

    def forward_qat(self, x: QTensor, dont_plot=False):
        """
        Collect stats, optionally fakequantize;
        per default also set qparams of monitored modules
        """
        # print(f"ql({self.qplotter.name}) input bits: {x.num_bits}; self.num_bits={self.num_bits}")
        super().forward_qat()
        assert isinstance(x, QTensor), type(x)
        if not self.distribution_kind==DistKind.CLIPPED:
            self._update_ema(x)

        if not self.dont_fakeQ:
            x = self.fake_quantize(
                x,
                self.quantization,
                self.num_bits,
                self.__stats__["min"],
                self.__stats__["max"],
                handling_qtensors=True,
            )
        if not self.set_qparams_during_quantization:
            # already make scale and zero accessible here for NonQuantizedModule
            # also done for KL_CLIPPED calibration in tensorRT's procedure
            self._set_qparams()

        if self.n_qat_batches == self.record_n_batches:
            self.freeze()
        assert not x.quantized
        x = self.qplotter(x, dont_plot=dont_plot)
        # print(f"ql({self.qplotter.name}) output bits: {x.num_bits}; self.num_bits={self.num_bits}")
        return x

    def forward_quantized(self, x: QTensor, dont_plot=False) -> QTensor:

        if __ASSERT__:
            # costly! remove these! TODO FIXME
            assert isinstance(x, QTensor)
            assert x.quantized
            assert is_integer(x._t)
            assert x._t.min() != x._t.max(), (x._t.min(), self.__stats__)
            assert len(torch.unique(x._t)) > 1, torch.unique(x._t)
        x = self.qplotter(x, dont_plot=dont_plot)
        return x

    def _update_ema(self, x: Union[Tensor, QTensor]):
        """
        Calculates EMA for activations, based on https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
        Each call calculates EMA for one module, specified by key.
        EMA statistics are saved to self.__stats__,
        a regular python dictionary containing EMA stats.
        If and what EMA stats are saved depends on self.calibration_mode and self.distribution_kind:
            * CalibMode.KL AND DistKind.SYMM: "mu"
            * CalibMode.KL AND DistKind.CLIPPED: nothing
            * CalibMode.EMA: "min", "max"
        :param x: QTensor, unless fp is True, then self.calibration_mode must be gaussian, and x FP32
        :param fp: string, name/identifier of current module
        :return: stats: updated EMA
        """
        assert not isinstance(x, Tensor), "REMOVE THIS"
        x = x.detach()
        assert not torch.isnan(x).any(), f"fraction of NaNs: {torch.isnan(x).sum().item()/x.nelement()}"

        if self.function is not None:
            # transform input by function (exp; for softmax)
            cache = x
            x = self.function(x)
            assert not torch.isinf(x).any(), f"fraction of infs: {torch.isinf(x).sum()/x.view(-1).shape[0]}, {cache.max().item()}, {cache.min().item()}"

        if self.distribution_kind == DistKind.SYMM:
            mu = torch.mean(x).item()
            # sigma = torch.std(x).item()
            dict_new_vals = {"mu": mu}
        else:
            assert self.calibration_mode == CalibMode.EMA
            min_val = torch.min(x).item()
            max_val = torch.max(x).item()
            assert max_val != min_val, (max_val, (x==max_val).all())
            dict_new_vals = {"min": min_val, "max": max_val}

        if not self.__stats__:
            self.__stats__ = dict_new_vals
        # else:
        #     for key, new_val in dict_new_vals.items():
        #         curr_val = self.__stats__[key]
        #         self.__stats__[key] = max(new_val, curr_val) if curr_val is not None else new_val

        for key, ema in self.__stats__.items():
            new_val = dict_new_vals[key]
            self.__stats__[key] -= (1 - self.ema_decay) * (ema - new_val)

    def quantize(self):
        if self.set_qparams_during_quantization:
            self._set_qparams()
        super().quantize()

    def _compute_thresholds(self, a, b, mu=None):
        if self.threshold_mode == ThresholdMode.Symmetric:
            return a, b
        elif self.threshold_mode == ThresholdMode.Independent:
            return a, b
        elif self.threshold_mode == ThresholdMode.Conjugate:
            assert mu is not None
            t = max(abs(mu - a), abs(mu - b))
            return (mu - t).item(), (mu + t).item()

    def _set_qparams(self):
        # sets quantization parameters of listened-to modules
        # depends on self.calibration_mode
        try:
            if self.calibration_mode==CalibMode.EMA:
                stats = self.__stats__
                scale, zero = self.quantization.calc_params(
                    min_val=stats["min"], max_val=stats["max"], num_bits=self.num_bits
                )
            else:
                if self.distribution_kind==DistKind.SYMM:
                    stats = self.__stats__
                    mu = stats["mu"]
                    a, b = self.hist_calib.compute_range(mu, title=self)
                    lower, upper = self._compute_thresholds(mu, a, b)
                    scale, zero = self.quantization.calc_params(
                        min_val=lower, max_val=upper, num_bits=self.num_bits
                    )
                elif self.distribution_kind==DistKind.CLIPPED:
                    a, b = self.hist_calib.compute_one_bound()
                    scale, zero = self.quantization.calc_params(
                        min_val=a, max_val=b, num_bits=self.num_bits
                    )
        except KeyError as KE:
            raise Exception(f"Got KeyError: {KE} during conversion of {self}. It was possibly never called during tuning?")

        prefix = self.name + "_" if self.name else ""

        # set attributes for all listened to modules and qlistener itself
        for mod in self.mods + [self]:
            msg = ("="*20)+"\n"+f"Successfully set {mod}'s qparams: {prefix}scale={scale}, {prefix}zero={zero}"
            setattr(mod, prefix+"scale", scale)
            setattr(mod, prefix+"zero", zero)
            if self.distribution_kind!=DistKind.CLIPPED:
                msg += f", {list(stats.keys())}"
                for key, ema in stats.items():
                    setattr(mod, prefix+key, ema)
            # qlogger.debug(msg+"\n"+("="*20))
        self._qparams_set = True

    def __repr__(self):
        s = f"QListener(\n"
        if self.name:
            s += "name={self.name},\n"
        s += "plot_name={self.plot_name},\n"
        s += "mods={self.mods},\n"
        s += "dist={self.distribution_kind},\n"
        s += "calib={self.calibration_mode})\n"
        s += ")"
        return s

class QPlotter(QuantizableModule):
    """
    Module to save histograms of forwarded Tensors;
    saving with different frequencies/indices per quantization stage.
    """
    def __init__(
            self,
            plot_name, # title of matplotlib plot
            stage_plot_freqs: Dict[str,int] = { # for each quantization stage, save a histogram every so often?
                "FP32": -1,
                "Calibration": -1,
                "QAT": -1,
                "Quantized": -1,
            },
            stage_plot_indices: Dict[str, List[int]] = { # for each quantization stage, save a histogram of the ith Tensor passing through?
                "FP32":[],
                "Calibration":[],
                "QAT":[],
                "Quantized":[],
            },
            **qkwargs
        ):
        super().__init__(**qkwargs)
        ext = "png" # save histogram as png files

        self.float_bins = qkwargs.get("calib_num_bins", 1000) # x axis resolution during floating point/calibration

        self.name = plot_name.replace(" ","_")
        self.stage_plot_freqs = stage_plot_freqs if \
                not "stage_plot_freqs" in qkwargs else qkwargs["stage_plot_freqs"]
        self.stage_plot_indices = stage_plot_indices if not "stage_plot_indices" in qkwargs else qkwargs["stage_plot_indices"]

        # create plot directory under name
        # FIXME TODO NOTE get log directory name and run name from TrainMan
        plots_dir = os.path.join("logs", qkwargs["name"], "plots")
        if not os.path.exists(plots_dir):
            os.mkdir(plots_dir)

        plot_dir = os.path.join(plots_dir, self.name)
        self.plot_dir = plot_dir # TODO add to cfg/CLI or read from datetime
        self.plot_tmpl = "{}_{}_{}."+ext

        # create a sub-plotdirectory for each quantization stage
        sub_dirs = [os.path.join(self.plot_dir, stage) for stage in list(self.stage_dict.values())]
        if not os.path.exists(self.plot_dir):
            os.mkdir(self.plot_dir)
            # qlogger.debug(f"Created directory {self.plot_dir}!")
        for sub_dir in sub_dirs:
            if not os.path.exists(sub_dir):
                os.mkdir(sub_dir)
                # qlogger.debug(f"Created directory {sub_dir}!")

        self.logger = logging.getLogger(name=self.name)
        # increase when dont_plot is not True in forward call
        self.stage_counters = {
            "FP32": 0,
            "Calibration": 0,
            "QAT": 0,
            "Quantized": 0,
        }

        # FOR MODEL GRAPH VISUALIZATION:
        latests_dir = os.path.join(plots_dir, "LATEST_PLOTS")
        if not os.path.exists(latests_dir):
            os.mkdir(latests_dir)

        # will create copy of file when plotting and write it to this file always:
        latest_copy = self.name+"_latest_{}." + ext
        self.latest_path = lambda: os.path.join(latests_dir, latest_copy.format(self.stage_str()))

        # encoding of path of copy of latest png for saving in backward context
        self.encoded_latest_copy = lambda: torch.Tensor(list(map(ord, latest_copy.format(self.stage_str()))))

        class PlotHackFn(torch.autograd.Function):
            """
            Hacky torch.autograd.Function used by tst_pytorch.QPlotter
            to save the path of the latest saved image.
            For visualizing the activations with pytorchviz
            """
            @staticmethod
            def forward(ctx, x: torch.Tensor):
                ctx.save_for_backward(self.encoded_latest_copy())
                return x

            @staticmethod
            def backward(ctx, grad_output):
                """ Identity """
                return grad_output

        self.plot_hack_fn = PlotHackFn.apply

    def forward_fp(self, x: Tensor, dont_plot=False) -> None:
        self._log(x, dont_plot=dont_plot)
        return self.plot_hack_fn(x)

    def forward_calib(self, x, *args, **kwargs):
        self._log(x, **kwargs)
        return self.plot_hack_fn(x)

    def forward_qat(self, x: Optional[QTensor] = None, dont_plot = False) -> QTensor:

        assert x.num_bits==self.num_bits, (x.num_bits, self.num_bits)
        self._log(x, dont_plot=dont_plot)
        out = self.plot_hack_fn(x._t)

        return QTensor(out, scale=x.scale, zero=x.zero, num_bits=self.num_bits, quantized=False)

    def forward_quantized(self, x: QTensor, dont_plot=False) -> QTensor:

        assert x.num_bits==self.num_bits, (x.num_bits, self.num_bits)
        self._log(x, dont_plot=dont_plot)
        out = self.plot_hack_fn(x._t)

        return QTensor(out, scale=x.scale, zero=x.zero, num_bits=self.num_bits, quantized=True)

    def _log(self, x: Union[Tensor, QTensor], dont_plot=False):
        """
        """

        bins = None
        if isinstance(x, QTensor):
            data, scale, zero = x._t, x.scale, x.zero
            if self.stage == QuantStage.QAT:
                # assert not x.quantized
                a = ( 0 - zero) * scale
                b = ((2**self.num_bits-1) - zero) * scale
                size = (b-a)/(2**self.num_bits-1)
            else:
                # data is quantized
                vmin, vmax = data.min(), data.max()

                through = len(torch.unique(data))
                mini = min(2**self.num_bits, data.nelement())
                through_ratio = (through/mini) * 100

                fstr = f"STATS:\nSCALE\t= {scale},\nZERO\t= {zero};"
                fstr += f"\nMIN\t= {vmin};\nMAX\t= {vmax};"
                fstr += f"\nSHAPE\t= {data.shape}"
                fstr += f"\nNELEM\t= {data.nelement()}"
                fstr += f"\nUNIQ\t= {data.unique()}"
                fstr += f"\n#UNIQ\t= {through} ({through_ratio}% of {mini})"

                self.logger.debug("="*20)
                self.logger.debug(fstr)

                assert x.quantized

                a = 0
                b = 2**self.num_bits-1
                size = 1

            # get bins
            bins = np.arange(a,b+size,size)
            info = f";\nscale={round(scale,3)}, zero={round(zero,3)}"
        else:
            bins = self.float_bins
            data = x
            info = ""

        if hasattr(self, "min"):
            assert hasattr(self, "max")
            info += f";\nmin={round(self.min, 2)}, max={round(self.max,2)}"

            # Original idea to investigate binning: Log Variance
            # sorted_data = data.reshape(-1).sort()[0]
            # shifts = (sorted_data - sorted_data.roll(1))
            # weighted_var = shifts.var()/(data.max()-data.min())
            # qlogger.debug(f"WEIGHTED_VAR({name}): {weighted_var.item()}")

        # if idx matches either of the two plotting criteria, store matplotlib histogram
        stage_str = self.stage_str()
        count = self.stage_counters[stage_str]
        freq = self.stage_plot_freqs[stage_str]
        plot_because_idx = count in self.stage_plot_indices[stage_str]
        plot_because_freq = freq > 0 and count % freq == 0

        if not dont_plot and (plot_because_idx or plot_because_freq):
            # plot data

            plot_data = data.detach().reshape(-1).cpu().numpy()

            plt.hist(plot_data, bins=bins)
            plt.ylabel("Frequency of bin")
            plt.title(stage_str+f" hist ({self.name}), batch #{count}"+info, fontsize=10)
            plt.gcf()

            # save
            fig_path = os.path.join(self.plot_dir, stage_str, self.plot_tmpl.format(stage_str, self.name, count))

            plt.savefig(fig_path)
            # update latest png (have to write png instead of symlink to above fig_path,
            # because graphviz considers symlinks unsafe)
            plt.savefig(self.latest_path())
            plt.gcf().clear()

        if not dont_plot:
            self.stage_counters[stage_str] += 1


# import these lists to files that define relevant modules (batchnorm.py)
# and add relevant modules to given list

# for calibration to work
# these must be updated in every module that adds QuantizableModules in need of a listener
global CLIPPING_MODULES, SYMMETRIZING_MODULES
CLIPPING_MODULES = [
    QReLU6,
    nn.ReLU6,
    nn.ReLU
] # ^ comprehensive list of modules that output clipped distribution

SYMMETRIZING_MODULES = [
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.Linear,
    QMatMul,
    QFill,
] # ^ comprehensive list of modules that output symmetric distribution


global OPS, NONQUANT
OPS = [
    nn.Conv2d,
    nn.Linear,
    nn.modules.batchnorm._BatchNorm,
    nn.MaxPool2d
]
NONQUANT = [
    NonQuantizableModuleWrap
]



