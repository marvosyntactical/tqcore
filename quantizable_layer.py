import torch
from torch import Tensor
from torch.fft import fft, fft2
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

from torch.nn.modules.utils import _pair

import math
import copy
from enum import Enum
from typing import Optional, Union, Tuple, Dict, Union, Callable
from collections import defaultdict
import warnings

from .qtensor import QTensor
from .quantization_functions import Quantization, \
        UniformQuantization, UniformSymmetricQuantization, FakeQuant
from .utils import print_qt_stats, is_integer, QuantConfigurationError
from .config import TuningMode, CalibMode, ThresholdMode, QuantStage, DistKind
from .histogram import HistogramCalibrator
from .kernel import qadd, qmul

__ASSERT__ = True

# this module contains quantizable versions of basic nn.Modules, as well as some helper modules
class QuantizableModule(nn.Module):
    """
    Interface for quantizable modules to implement.

    During fp training, this module acts as Identity if forward is not overwritten.
    It also has a Tuning Stage, which can be either
        * Quantization Aware Training (QAT) or
        * Calibration
    and then a fully quantized stage;
    Each of the stages should be implemented by subclasses.
    """
    FLOAT_SCALE = 1. # this determines all successive scales
    FLOAT_ZERO = 0.

    # initialize these these default values
    # so during very first QAT forward pass, modules can access these attributes.
    scale_next = FLOAT_SCALE
    zero_next = FLOAT_ZERO
    # on successive QAT forward passes, QListener manages these values

    def __init__(
            self,
            quantization: Quantization = UniformQuantization,
            weight_quantization: Quantization = UniformSymmetricQuantization,
            num_bits: int = 8,
            num_bits_weight: int = 8,
            num_bits_bias: int = 32,
            nudge_zero: bool = False,
            **qkwargs,
        ):
        nn.Module.__init__(self) # super().__init__ somehow calls __init__ of _QBatchNorm for subclasses of it FIXME

        self.num_bits = num_bits
        self.quantization = quantization(nudge_zero=nudge_zero)
        self.num_bits_weight = num_bits_weight
        self.weight_quantization = weight_quantization(nudge_zero=nudge_zero)
        self.num_bits_bias = num_bits_bias

        self.stage = QuantStage.FP32
        self.forward = self.forward_fp # changes from stage to stage

    def forward_fp(self, x: Tensor) -> Tensor:
        return x

    def forward_qat(self, x: QTensor) -> QTensor:
        raise NotImplementedError(f"{type(self)}.forward_qat")

    def forward_quantized(self, x: QTensor) -> QTensor:
        raise NotImplementedError(f"{type(self)}.forward_quantized")

    def forward_calib(self, *args, **kwargs):
        return self.forward_fp(*args, **kwargs)

    def calibration_prepare(self):
        self.stage = QuantStage.Calibration
        self.forward = self.forward_calib

    def qat_prepare(self):
        self.stage = QuantStage.QAT
        self.forward = self.forward_qat

    def quantize(self):
        self.stage = QuantStage.Quantized
        self.forward = self.forward_quantized

        if not [attr for attr in dir(self) if "scale" in attr or "zero" in attr]:
            warnings.warn(
                f"""
                During {self}.quantize(), no scale or zero attribute were found.
                These should be set for this instance of {type(self)}.
                Could be due to calling super().quantize() before setting self.zero/self.scale.
                """
            )

class QuantStub(QuantizableModule):
    """
    Quantizes incoming torch.Tensors into tqcore.QTensors if necessary.
    Functionality analogous to torch.quantization.QuantStub
    """

    def forward_qat(self, x):
        # no affine transformation
        if isinstance(x, Tensor)\
                and torch.is_floating_point(x)\
                and not isinstance(x, QTensor):
            r = QTensor(
                x,
                scale=self.FLOAT_SCALE,
                zero=self.FLOAT_ZERO,
                quantized=False
            )
        return r

    def forward_quantized(self, x):
        # affine transformation to learned range
        if isinstance(x, Tensor)\
                and torch.is_floating_point(x)\
                and not isinstance(x, QTensor):
            r = self.quantization.quantize_to_qtensor_using_params(
                x,
                self.scale,
                self.zero,
                num_bits=self.num_bits,
            )
        return r

class DeQuantStub(QuantizableModule):
    """
    Dequantizes incoming torch.Tensors into tqcore.QTensors if necessary.
    Functionality Analogous to torch.quantization.DeQuantStub
    """
    def __init__(self, **qkwargs):
        super().__init__(**qkwargs)

    def _process_outputs(self, outputs, f):
        if not isinstance(outputs, tuple):
            assert isinstance(outputs, QTensor), type(outputs)
            outputs = (outputs,)

        outputs = list(outputs)

        for i, out in enumerate(outputs):
            if isinstance(out, QTensor):
                outputs[i] = f(out)

        if len(outputs) == 1:
            outputs = outputs[0]
        else:
            outputs = tuple(outputs)
        return outputs

    def forward_qat(self, outputs: Union[QTensor, Tuple[QTensor]]):
        f = lambda qt: qt._t
        return self._process_outputs(outputs, f)

    def forward_quantized(self, outputs: Union[QTensor, Tuple[QTensor]]):
        f = lambda qt: qt.dequantize()
        return self._process_outputs(outputs, f)

class QMul(QuantizableModule):
    def forward_fp(self, a: torch.Tensor, b: torch.Tensor):
        return a * b

    def forward_qat(self, a: QTensor, b: QTensor):
        # NO affine transformation; multiply in float but keep track of scale
        rfloat = a._t * b._t
        r = QTensor(rfloat, scale=self.scale, zero=self.zero, quantized=False)
        return r

    def forward_quantized(
            self,
            a: Union[QTensor, torch.Tensor, nn.Parameter, float, int],
            b: Union[QTensor, torch.Tensor, nn.Parameter, float, int],
        ):
        # affine transformation; simulates low bit multiplication
        return qmul(a, b, 1., self.scale, self.zero, torch.mul,
                self.quantization, self.weight_quantization,
                self.num_bits, self.num_bits_weight)


class QMatMul(QuantizableModule):
    def __init__(self, *args, factor: float=1., **qkwargs):
        super().__init__(*args, **qkwargs)
        self.factor = float(factor)

    def forward_fp(self, a, b):
        return self.factor * ( a @ b )

    def forward_qat(self, a, b):
        rfloat = self.factor * a._t @ b._t
        r = QTensor(rfloat, scale=self.scale, zero=self.zero, quantized=False)
        return r

    def forward_quantized(self, a: QTensor, b:QTensor) -> QTensor:
        return qmul(
                a, b, self.factor, self.scale, self.zero, torch.matmul,
                self.quantization, self.weight_quantization,
                self.num_bits, self.num_bits_weight)

class QAdd(QuantizableModule):
    def __init__(self, *args, **qkwargs):
        super().__init__(*args, **qkwargs)

    def forward_fp(self, a, b):
        return a + b

    def forward_qat(self, a: QTensor, b: QTensor):
        rfloat = a._t + b._t
        r = QTensor(rfloat, scale=self.scale, zero=self.zero, quantized=False)
        return r

    def forward_quantized(self, a: QTensor, b:QTensor) -> QTensor:
        print("start qt stats in QAdd:")
        print_qt_stats("left", a)
        print_qt_stats("right", b)
        print("end qt stats in QAdd:")
        return qadd(
            a, b, 1.,
            self.scale, self.zero, torch.add,
            self.quantization, self.weight_quantization,
            self.num_bits, self.num_bits_weight
        )

class QMask(QuantizableModule):
    def __init__(self, fp_neg_val: float=-1e5, **qkwargs):
        super().__init__(**qkwargs)
        self.fp_neg_val = float(fp_neg_val)

    def forward_fp(self, scores, mask):
        scores = scores.masked_fill(mask==torch.as_tensor(False), self.fp_neg_val)
        return scores

    def forward_qat(self, scores, mask):
        scores = scores._t.masked_fill(mask==torch.as_tensor(False), self.fp_neg_val)
        return QTensor(scores, self.scale, self.zero, quantized=False)

    def forward_quantized(self, scores, mask):
        scores = scores.masked_fill(mask==torch.as_tensor(False), self.zero)
        return QTensor(scores, self.scale, self.zero)

class QSoftmax(QuantizableModule):

    def __init__(self, dim=-1, time_window: int = 144, alpha:float = None, **qkwargs):
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
            name="exp",
            function=self._scaled_exp,
            dont_fakeQ=True,
            **qkwargs
        )

        # this ALSO records stats of normed output (which will be between 0 and 1 anyway),
        # but is used mainly because of fake quantizing
        self.norm_listener = QListener(self, name="normed", **qkwargs)

    def _scaled_exp(self, inp: torch.Tensor):
        # (differentiable)
        exponent = self.alpha * inp
        # out = torch.exp(exponent.clamp(max=78.))
        out = torch.exp(exponent)

        assert not torch.isinf(out).sum(), f"QSoftmax._scaled_exp produced a ratio of {torch.isinf(out).sum()/(out==out).sum()} infs"
        return out

    def _set_exp_lkp(self):
        # this range is tailored to uniformquantization
        self.EXP_STEP_FUN = self._scaled_exp(torch.arange(2.**self.num_bits)).round()

    def _exp_lkp(self, long_tensor: torch.LongTensor):
        # (not differentiable)
        return QTensor(
            self.EXP_STEP_FUN[long_tensor], scale=self.exp_scale, zero=self.exp_zero,
            quantized=False
        )

    def forward_fp(self, inp: torch.Tensor) -> torch.Tensor:
        out = inp.softmax(dim=self.dim)
        # print("Softmax return:")
        # print(torch.unique(out))
        return out

    def forward_qat(self, inp: torch.Tensor) -> torch.Tensor:
        self.exp_listener(inp)

        # LogSumExp trick:
        numerator = self._scaled_exp(inp-inp.max())
        out = numerator/numerator.sum(dim=self.dim).unsqueeze(self.dim)

        out = self.norm_listener(out)

        # print("Softmax return:")
        # print(torch.unique(out))
        return out

    def quantize(self):
        super().quantize()
        self._set_exp_lkp()

    def forward_quantized(self, inp: QTensor) -> QTensor:

        print(f"Softmax quantized fwd debug:")
        print(self.exp_scale, self.exp_zero)
        print(self.normed_scale, self.normed_zero)
        print(f"mean of inp: {inp._t.mean().item()} (if high, logsumexp can help!)")
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^")

        exponentiated: QTensor = self._exp_lkp(inp._t.long()) # NOTE: range from 0 to 255 here already!
        zeroed_exp =  exponentiated._t - exponentiated.zero
        M = exponentiated.scale / (self.normed_scale * self.alpha)
        normed_exp = (zeroed_exp * M) + self.normed_zero
        r = self.quantization.tensor_clamp(normed_exp, num_bits=self.num_bits)

        # print("Softmax return:")
        # print(torch.unique(r))

        # NOTE: v Dequantized Implementation, leads to random performance after quantizing
        # r: torch.Tensor = (QTensor(inp._t * self.alpha, scale=inp.scale, zero=inp.zero)).dequantize().softmax(dim=self.dim)

        # r: QTensor = self.quantization.quantize_to_qtensor_using_params(
        #     r,
        #     self.normed_scale_next,
        #     self.normed_zero_next,
        #     num_bits=self.num_bits
        # )

        # denominator = self.quantization.quantize_to_qtensor_using_params(
        #     1/exponentiated.sum(dim=self.dim).unsqueeze(-1),
        #     scale=1/self.exp_scale_next,
        #     zero=self.exp_zero_next,
        #     num_bits=self.num_bits
        # )

        # print("denominator: scale=",denominator.scale, "zero=",denominator.zero)

        # r = qmul(
        #     numerator, denominator, 1.,
        #     self.normed_scale_next, self.normed_zero_next,
        #     torch.mul,
        #     self.quantization, self.weight_quantization,
        #     self.num_bits, self.num_bits_weight
        # )

        return r

class NonQuantizableModuleWrap(QuantizableModule):

    def __init__(self, module, *args, **qkwargs):
        super().__init__(**qkwargs)

        self.fp_module = module

        self.in_listener = QListener(
            self,
            name="in",
            dont_fakeQ=True,
            **qkwargs
        )

        self.out_listener = QListener(
            self,
            name="out",
            dont_fakeQ=False,
            **qkwargs
        )

    def forward_fp(self, *args, **kwargs) -> torch.Tensor:
        r = self.fp_module(*args, **kwargs)
        return r

    def forward_qat(self, *args, **kwargs) -> torch.Tensor:

        fp_args = [a._t if isinstance(a, QTensor) else a for a in args]

        fp_out = self.fp_module(*fp_args, **kwargs)

        if not isinstance(fp_out, tuple):
            fp_out = (fp_out,)
        fakeq_out = [
            self.out_listener(
                QTensor(fp_o, self.FLOAT_SCALE, self.FLOAT_ZERO, quantized=False)) \
                    if isinstance(fp_o, Tensor)
                    else fp_o
            for fp_o in fp_out
        ]

        return tuple(fakeq_out) if len(fakeq_out) > 1 else fakeq_out[0]

    def forward_quantized(self, inp: QTensor, *args, **kwargs) -> QTensor:
        print(f"nonQ QAT params: {inp.zero}, {inp.scale}")

        fp_args, is_q = zip(*[
            (a.dequantize(), True) if isinstance(a, QTensor) else (a, False) \
        for a in args])

        fp_outs = self.fp_module(*fp_args, **kwargs)

        q_outs = [
        self.quantization.quantize_to_qtensor_using_params(
            fp_out,
            scale=self.out_scale,
            zero=self.out_zero,
            num_bits=self.num_bits
        ) if is_q[i] else fp_out for i, fp_out in enumerate(fp_outs)]

        return tuple(q_outs) if len(q_outs) > 1 else q_outs[0]

class QReLU6(QuantizableModule):

    def forward_fp(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.relu6(x)

    def forward_qat(self, x: torch.Tensor) -> torch.Tensor:

        scale, zero = self.scale, self.zero

        six = round(6 / scale + zero)
        out = x.clamp(min=zero, max=six)
        # followed by fakeQuantization
        return out

    def forward_quantized(self, x: QTensor) -> QTensor:

        scale = self.scale
        zero = self.zero
        six = round(6 / scale + zero)
        out = x._t.clamp(min=zero, max=six)

        assert round(scale, 5) == round(x.scale, 5), \
                (scale, x.scale)
        assert round(zero, 5) == round(x.zero, 5), \
                (zero, x.zero)

        return QTensor(out, scale=scale, zero=zero)

def DiscreteHartleyTransform(input):
    # from https://github.com/AlbertZhangHIT/Hartley-spectral-pooling/blob/master/spectralpool.py
    # TODO test as alternative
    fft = torch.rfft(input, 2, normalized=True, onesided=False)
    dht = fft[:, :, :, :, -2] - fft[:, :, :, :, -1]
    return dht

class FFT(nn.Module):
    # TODO move this to ..modules.py
    # TODO figure out mask

    # figure out whether to orthonormalize (scale by 1/sqrt(n))
    # paper: vandermonde matrix has normalization
    # third party code: no normalization

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):

        # x = fft(fft(x, dim=-1), dim=-2).real
        x = fft2(x)
        x = x.real #  + x.imag
        return x

class QFFT(QuantizableModule):
    # TODO figure out mask

    # figure out whether to use kwarg "normalize" (scale by 1/sqrt(n))
    # paper: vandermonde matrix has normalization
    # third party code: no normalization

    def __init__(self, *args, **kwargs):
        assert False, NotImplemented
        super().__init__()

    def forward_fp(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        assert False, NotImplemented

        # x = fft(fft(x, dim=-1), dim=-2).real
        x = fft2(x)
        x = x.real #  + x.imag
        return x

    def forward_qat(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        assert False, NotImplemented
        x = fft2(x)
        x = x.real #  + x.imag
        return x

    def forward_quantized(self, x: QTensor, mask: Optional[torch.Tensor] = None):
        assert False, NotImplemented
        x = fft2(x)
        x = x.real #  + x.imag
        return x

class QPositionalEncoding(QuantizableModule):
    """
    Learnable Position Encoding (A W x D bias matrix that we add to the input)
    """
    def __init__(self,
                 dim: int = 0,
                 time_window: int = 24,
                 init_fn: Callable = torch.rand
                 ):
        super().__init__()

        self.W = nn.Parameter(init_fn(time_window, dim))

    def forward_fp(self, X: torch.Tensor):
        """
        Encode inputs.
        Args:
            X (FloatTensor): Sequence of word vectors
                ``(batch_size, dim, time_window)``
        """
        # Add position encodings
        out = self.W + X
        return out

    def forward_quantized(self, X: QTensor):
        return qadd(
            self.W, X, 1.,
            self.scale, self.zero, torch.add,
            self.quantization, self.weight_quantization,
            self.num_bits, self.num_bits_weight
        )


class QListener(QuantizableModule):
    """
    During qat, this module records min, max values of torch.Tensor s passing through.
    (no other module records stats)
    Constructed with n nn.Modules "modules", for each of which the QListener sets the attributes:
        - module.scale_next
        - module.zero_next
        - module.min
        - module.max
    Or, optionally, module.name_scale_next, and so on.
    """
    def __init__(
            self,
            *modules: nn.Module,
            name = None,
            function = None,
            dont_fakeQ: bool = False,
            ema_decay: float = .9999,
            nudge_zero: bool = False,
            calibration: Optional[str] = None, # manually override qkwargs["calib_mode"]
            clipped_distr: Optional[bool] = None, # manually override distribution type
            **qkwargs
        ):

        super().__init__(**qkwargs)

        self.function = function # optionally apply function before collecting stats (for softmax)
        self.name = "" if name is None else str(name) # set attribute name_scale_next and so on
        self.ema_decay = ema_decay
        self.set_qparams_during_quantization = False # updates range after every tuning forward call
        self.mods = list(modules)

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
                        # assume clipping is applied to everything
                        warnings.warn(f"QListeneder encountered some clipping, some symmetrizing input modules. Assuming everything is clipped.")
                        self.distribution_kind = DistKind.CLIPPED
                        distkind = "clipped"
                warnings.warn(f"QListener listening to\n{self.mods}\ndecided on {distkind}")
        else:
            # distribution does not matter if we just estimate EMA min/max
            if clipped_distr is not None:
                warnings.warn(f"QListener listening to\n{self.mods}\nignores given kwarg clipped_distr={clipped_distr} because it calibrates with EMA only")
            self.distribution_kind = DistKind.IGNORE

        if self.distribution_kind != DistKind.CLIPPED:
            self.__stats__ = defaultdict()
            # TODO FIXME is this still necessary?
            for module in self.mods:
                setattr(module, self.name+'__stats__', self.__stats__)

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
        if not dont_fakeQ:
            self.fake_quantize = FakeQuant.apply_wrapper

        self._qparams_set = False

    def forward_calib(self, x: Tensor):
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
        return x

    def forward_qat(self, x: QTensor):
        """
        Collect stats and possibly fakequantize
        """
        if not self.distribution_kind==CalibMode.CLIPPED:
            self._update_ema(x)

        if not self.dont_fakeQ:
            scale, zero = self.quantization.calc_params(
                self.__stats__["min"],
                self.__stats__["max"],
                num_bits=self.num_bits
            )

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
        return x

    def forward_quantized(self, x: QTensor) -> QTensor:

        if __ASSERT__:
            # costly! remove these! TODO FIXME
            assert isinstance(x, QTensor)
            assert is_integer(x._t)
            assert x._t.min() != x._t.max(), (x._t.min(), self.__stats__)
            assert len(torch.unique(x._t)) > 1, torch.unique(x._t)
        return x

    def _update_ema(self, x: QTensor=None):
        """
        Calculates EMA for activations, based on https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
        Each call calculates EMA for one layer, specified by key.
        EMA statistics are saved to self.__stats__,
        a regular python dictionary containing EMA stats.
        If and what EMA stats are saved depends on self.calibration_mode and self.distribution_kind:
            * CalibMode.KL and DistKind.SYMM: "mu"
            * CalibMode.EMA: "min", "max"
        :param x: QTensor, unless fp is True, then self.calibration_mode must be gaussian, and x FP32
        :param fp: string, name/identifier of current layer
        :return: stats: updated EMA
        """
        x = x.detach()
        assert not torch.isnan(x).any(), f"fraction of NaNs: {torch.isnan(x).sum()/x.view(-1).shape[0]}"

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
        else:
            for key, new_val in dict_new_vals.items():
                curr_val = self.__stats__[key]
                self.__stats__[key] = max(new_val, curr_val) if curr_val is not None else new_val

        for key, ema in self.__stats__.items():
            new_val = dict_new_vals[key]
            self.__stats__[key] -= (1 - self.ema_decay) * (ema - new_val)

        for module in self.mods:
            setattr(module, self.name+'__stats__', self.__stats__)

    def quantize(self):
        if self.set_qparams_during_quantization:
            self._set_qparams()
        return super().quantize()

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
        # depends on self.calibration_mode
        try:
            if self.calibration_mode==CalibMode.EMA:
                stats = self.__stats__
                scale, zero = self.quantization.calc_params(
                    stats["min"], stats["max"], num_bits=self.num_bits
                )
            else:
                if self.distribution_kind==DistKind.SYMM:
                    stats = self.__stats__
                    mu = stats["mu"]
                    a, b = self.hist_calib.compute_range(mu, title=self)
                    lower, upper = self._compute_thresholds(mu, a, b)
                    scale, zero = self.quantization.calc_params(
                        lower, upper, num_bits=self.num_bits
                    )
                elif self.distribution_kind==DistKind.CLIPPED:
                    a, b = self.hist_calib.compute_one_bound()
                    scale, zero = self.quantization.calc_params(
                        a, b,  num_bits=self.num_bits
                    )
        except KeyError as KE:
            raise Exception(f"Got KeyError: {KE} during QListener (listening to {self.mods}) conversion. It was possibly never called during tuning?")

        prefix = self.name + "_" if self.name else ""


        # set attributes for all listened modules
        for mod in self.mods:
            msg = ("="*20)+"\n"+f"Successfully set {mod}'s qparams: scale, zero"
            setattr(mod, prefix+"scale", scale)
            setattr(mod, prefix+"zero", zero)
            if self.distribution_kind!=DistKind.CLIPPED:
                msg += f", {list(stats.keys())}"
                for key, ema in stats.items():
                    setattr(mod, prefix+key, ema)
            print(msg+"\n"+("="*20))
        self._qparams_set = True

    def __repr__(self):
        s = f"QListener(mods={self.mods}, dist={self.distribution_kind}, calib={self.calibration_mode})"
        return s

# these must be updated in every module that adds QuantizableModules that belong here
global CLIPPING_MODULES, SYMMETRIZING_MODULES
CLIPPING_MODULES = [
    QReLU6,
    nn.ReLU6,
    nn.ReLU
] # comprehensive list of modules that output clipped distribution
SYMMETRIZING_MODULES = [
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.Linear
]

