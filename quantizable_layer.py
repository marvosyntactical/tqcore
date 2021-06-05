import torch
from torch import Tensor
from torch.fft import fft, fft2
import torch.nn as nn
import torch.nn.functional as F

from .qtensor import QTensor
from .quantization_functions import Quantization, \
        UniformQuantization, UniformSymmetricQuantization, FakeQuant

from torch.nn.modules.utils import _pair

import math
import copy
from typing import Optional, Union
from functools import partial

__DEBUG__ = True
is_integer = lambda t: ((t.round()==t).all() if t.shape else t.round()==t) if __DEBUG__ else True

# this module contains quantizable versions of basic nn.Modules, as well as some helper modules

class QuantizableModule(nn.Module):
    """
    Interface for quantizable modules to implement.

    During fp training, this module acts as Identity.
    It also has a Quantization Aware Training (QAT) stage, and quantized stage;
    Each of the three should be implemented by subclasses.
    (otherwise this module remains an Identity)
    """

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

        self.forward = self.forward_fp

    def forward_fp(self, x: Tensor) -> Tensor:
        return x

    def forward_qat(self, x: Tensor) -> Tensor:
        raise NotImplementedError(f"{type(self)}.forward_qat")

    def forward_quantized(self, x: QTensor) -> QTensor:
        raise NotImplementedError(f"{type(self)}.forward_quantized")

    def qat_prepare(self):
        self.forward = self.forward_qat

    def quantize(self):
        self.forward = self.forward_quantized


class Quant(QuantizableModule):
    """
    Quantizes incoming torch.Tensors into tqcore.QTensors if necessary.
    Functionality Analogous to torch.quantization.QuantStub
    """

    def forward_qat(self, x):
        return x

    def forward_quantized(self, x):
        if isinstance(x, Tensor)\
                and torch.is_floating_point(x)\
                and not isinstance(x, QTensor):
            r = self.quantization.quantize_to_qtensor(
                x,
                self.min_val,
                self.max_val,
                num_bits=self.num_bits
            )
        return r

class DeQuant(QuantizableModule):
    """
    Dequantizes incoming torch.Tensors into tqcore.QTensors if necessary.
    Functionality Analogous to torch.quantization.QuantStub
    """
    def __init__(self, **qkwargs):
        super().__init__(**qkwargs)

    def forward_qat(self, x):
        return x

    def forward_quantized(self, outputs):

        if not isinstance(outputs, tuple):
            assert isinstance(outputs, QTensor), type(outputs)
            outputs = (outputs,)

        outputs = list(outputs)

        for i, out in enumerate(outputs):
            if isinstance(out, QTensor):
                outputs[i] = out.dequantize()

        if len(outputs) == 1:
            outputs = outputs[0]
        else:
            outputs = tuple(outputs)
        return outputs

class QListener(QuantizableModule):
    """
    During qat, this module records min, max values of torch.Tensor s passing through.
    (no other module records stats)
    Accepts an iterable of modules for each of which the QListener sets the module.scale_next attribute and so on
    """
    def __init__(self, *modules: nn.Module, name = None, function = None, dont_fakeQ: bool = False, ema_decay: float = .9999, nudge_zero: bool = False, **qkwargs):

        super().__init__(**qkwargs)

        self.function = function # optionally apply function before collecting stats (for softmax)
        self.name = "" if name is None else str(name) # set attribute name_scale_next and so on
        self.ema_decay = ema_decay

        self.__stats__ = {}
        for module in modules:
            exec(f"module.{self.name+'__stats__'} = self.__stats__")
        self.mods = list(modules)

        self.dont_fakeQ = dont_fakeQ
        if not dont_fakeQ:
            self.fake_quantize = FakeQuant.apply

    def forward_qat(self, x: Tensor):
        """
        Collect stats AND fakequantize
        """
        self._update_ema_stats(x)

        if not self.dont_fakeQ:
            scale, zero = self.quantization.calc_params(
                self.__stats__["ema_min"], self.__stats__["ema_max"],
                num_bits=self.num_bits
            )

            x = self.fake_quantize(
                x,
                self.quantization,
                self.num_bits,
                self.__stats__["ema_min"],
                self.__stats__["ema_max"]
            )
        print("listener qat output type: ",type(x))
        return x

    def forward_quantized(self, x: QTensor) -> QTensor:
        assert isinstance(x, QTensor)
        assert is_integer(x._t)
        assert x._t.min() != x._t.max(), (x._t.min(), self.__stats__)
        assert len(torch.unique(x._t)) > 1, torch.unique(x._t)
        print("asserted stuff;", len(torch.unique(x._t)))
        return x

    def _update_ema_stats(self, x):
        """
        Calculates EMA/ MA for activations, based on https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
        Each call calculates EMA for one layer, specified by key.
        Stats is a regular python dictionary containing EMA for multiple layers.
        :param x: activation tensor of layer
        :param stats: dictionary: EMA
        :param key: string, name/identifier of current layer
        :return: stats: updated EMA
        """
        x = x.detach()

        if self.function is not None:
            cache = x
            x = self.function(x)
            assert not torch.isinf(x).any(), f"fraction of infs: {torch.isinf(x).sum()/x.view(-1).shape[0]}, {cache.max().item()}, {cache.min().item()}"

        max_val = torch.max(x).item()
        min_val = torch.min(x).item()

        assert not torch.isnan(x).any(), f"fraction of NaNs: {torch.isnan(x).sum()/x.view(-1).shape[0]}"
        # assert max_val != min_val, (max_val, (x==max_val).all())

        if not self.__stats__:
            self.__stats__ = {"ema_max": max_val, "ema_min": min_val}

        curr_max = self.__stats__["ema_max"]
        curr_min = self.__stats__["ema_min"]
        self.__stats__['ema_max'] = max(max_val, curr_max) if curr_max is not None else max_val
        self.__stats__['ema_min'] = max(min_val, curr_min) if curr_min is not None else min_val


        if 'ema_min' in self.__stats__:
            # self.__stats__['ema_min'] = (1.-self.ema_decay) * min_val + self.ema_decay * self.__stats__['ema_min']
            self.__stats__['ema_min'] -=  (1 - self.ema_decay) * (self.__stats__['ema_min'] - min_val)
        else:
            self.__stats__['ema_min'] = min_val

        if 'ema_max' in self.__stats__:
            # self.__stats__['ema_max'] = (1.-self.ema_decay) * max_val + self.ema_decay * self.__stats__['ema_max']
            self.__stats__['ema_max'] -= (1 - self.ema_decay) * (self.__stats__['ema_max'] - max_val)
        else:
            self.__stats__['ema_max'] = max_val

        # assert not torch.isnan(torch.Tensor([self.__stats__["ema_max"]]))
        # assert not torch.isnan(torch.Tensor([self.__stats__["ema_min"]]))

        # FIXME need to update every module's stats individually
        # previously, it worked to just set their stats to another module's stats dict
        # and update that. Now the id()s change when doing that and updating only
        # the QListener dict. So now I've got to have a list with all the modules,
        # which may be dangerous with certain other things
        for module in self.mods:
            exec(f"module.{self.name+'__stats__'} = self.__stats__")

    def quantize(self):
        super().quantize()

        stats = self.__stats__

        try:
            scale_next, zero_next = self.quantization.calc_params(
                stats["ema_min"], stats["ema_max"], num_bits=self.num_bits
            )
        except KeyError as KE:
            if True in [isinstance(mod, QMask) or isinstance(mod, QSoftmax) for mod in self.mods]:
                print(f"Got KeyError: {KE} during QListener conversion.")
                return
            else:
                raise

        prefix = self.name + "_" if self.name else ""

        for mod in self.mods:
            setattr(mod, prefix+"scale_next", scale_next)
            setattr(mod, prefix+"zero_next", zero_next)
            setattr(mod, prefix+"min_val", stats["ema_min"])
            setattr(mod, prefix+"max_val", stats["ema_max"])


def _qmul(
        a: QTensor, b: QTensor, factor: float,
        scale_next, zero_next, op,
        quantization, weight_quantization,
        num_bits, num_bits_weight
    ) -> QTensor:
    # helper func for mul and matmul
    # TODO future:
    # replace this and QAdd.forward_quantized by gemmlowp (possibly <) 8 bit kernel

    ab = [a,b]
    for i, t in enumerate(ab):
        if not isinstance(t, QTensor):
            if isinstance(t, torch.Tensor):
                if isinstance(t, nn.Parameter):
                    # e.g. elementwise mul with parameter
                    t = weight_quantization.quantize_to_qtensor(
                        t,
                        min_val=t.min().item(),
                        max_val=t.max().item(),
                        num_bits=num_bits_weight,
                    )
                else:
                    # e.g. rescale in mhattn
                    t = quantization.quantize_to_qtensor(
                        t,
                        min_val=t.min().item(),
                        max_val=t.max().item(),
                        num_bits=num_bits,
                    )
            else:
                assert False, (t, type(t))
                # t = QTensor(torch.as_tensor(t), scale=1., zero=0.)

        ab[i] = t

    a, b  = ab

    a_zeroed = a._t - round(a.zero)
    b_zeroed = b._t - round(b.zero)

    r: torch.Tensor = op(a_zeroed, b_zeroed)
    r_float = r

    multiplier = (a.scale * b.scale * factor) / scale_next
    # scale result tensor back to given bit width, saturate to uint if unsigned is used:
    r = r * multiplier + zero_next
    r_unclamped = r

    # round and clamp
    r = quantization.tensor_clamp(r, num_bits=num_bits)

    # Make sure we didnt move outside of EMA range:
    assert r.min() != r.max(), \
        (scale_next, zero_next, r.min(), a._t.min(), a._t.max(), b._t.min(), b._t.max(), r_float.min(), r_float.max(), r_unclamped.min(), r_unclamped.max(), multiplier, factor)

    return QTensor(r, scale=scale_next, zero=zero_next)

class QMul(QuantizableModule):
    def forward_fp(self, a: torch.Tensor, b: torch.Tensor):
        return a * b

    def forward_qat(self, a: torch.Tensor, b: torch.Tensor):
        return a * b

    def forward_quantized(
            self,
            a: Union[QTensor, torch.Tensor, nn.Parameter, float, int],
            b: Union[QTensor, torch.Tensor, nn.Parameter, float, int],
        ):
        return _qmul(a, b, 1., self.scale_next, self.zero_next, torch.mul,
                self.quantization, self.weight_quantization,
                self.num_bits, self.num_bits_weight)


class QMatMul(QuantizableModule):
    def __init__(self, *args, factor: float=1., **qkwargs):
        super().__init__(*args, **qkwargs)
        self.factor = float(factor)

    def forward_fp(self, a, b):
        return self.factor * ( a @ b )

    def forward_qat(self, a, b):
        return self.factor * ( a @ b )

    def forward_quantized(self, a: QTensor, b:QTensor) -> QTensor:
        return _qmul(
                a, b, self.factor, self.scale_next, self.zero_next, torch.matmul,
                self.quantization, self.weight_quantization,
                self.num_bits, self.num_bits_weight)

class QAdd(QuantizableModule):
    def __init__(self, *args, **qkwargs):
        super().__init__(*args, **qkwargs)

    def forward_fp(self, a, b):
        return a + b

    def forward_qat(self, a, b):
        return a + b

    def forward_quantized(self, a: QTensor, b:QTensor) -> QTensor:
        # wrapped version of the earlier "globalparams" implementation:
        # https://cegit.ziti.uni-heidelberg.de/mkoss/tqcore/-/blob/globalparams/quantized_layer.py#L206

        a_requantized = self.quantization.quantize_to_qtensor_using_params(
            a.dequantize(),
            scale=1/(.5*self.scale_next),
            zero=.5*self.zero_next,
            num_bits=self.num_bits-1
        )
        b_requantized = self.quantization.quantize_to_qtensor_using_params(
            b.dequantize(),
            scale=1/(.5*self.scale_next),
            zero=.5*self.zero_next,
            num_bits=self.num_bits-1
        )

        r = a_requantized + b_requantized

        assert is_integer(r._t), r

        return r

class QMask(QuantizableModule):
    def __init__(self, fp_neg_val: float=-1e5, **qkwargs):
        super().__init__(**qkwargs)
        self.fp_neg_val = float(fp_neg_val)

    def forward_fp(self, scores, mask):
        scores = scores.masked_fill(mask==torch.as_tensor(False), self.fp_neg_val)
        return scores

    def forward_qat(self, scores, mask):
        scores = scores.masked_fill(mask==torch.as_tensor(False), self.fp_neg_val)
        return scores

    def forward_quantized(self, scores, mask):
        scores = scores.masked_fill(mask==torch.as_tensor(False), self.zero_next)
        return scores

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
        return QTensor(self.EXP_STEP_FUN[long_tensor], scale=self.exp_scale_next, zero=self.exp_zero_next)

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
        print(self.exp_scale_next, self.exp_zero_next)
        print(self.normed_scale_next, self.normed_zero_next)
        print(f"mean of inp: {inp._t.mean().item()} (if high, logsumexp can help!)")
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^")

        exponentiated: QTensor = self._exp_lkp(inp._t.long()) # NOTE: range from 0 to 255 here already!

        zeroed_exp =  exponentiated._t - exponentiated.zero
        M = exponentiated.scale / (self.normed_scale_next * self.alpha)
        normed_exp = (zeroed_exp * M) + self.normed_zero_next
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

        # r = _qmul(
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
            **qkwargs
        )

    def forward_fp(self, inp: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.fp_module(inp, *args, **kwargs)

    def forward_qat(self, inp: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # dequantize fake quantized torch.Tensor, do fp fwd pass, and re-fakequantize

        # FIXME FIXME FIXME TODO FIXME FIXME FIXME
        # assert False, f"fix fwd qat of nonquantmodule: dont know how to rescale tensor to be of float scale"
        print(inp, inp.dtype)

        fakeq_inp = self.in_listener(inp)

        # min_val = self.in_listener.__stats__.get("ema_min")
        # max_val = self.in_listener.__stats__.get("ema_max")

        # in_scale, in_zero = self.quantization.calc_params(
        #     min_val, max_val, num_bits=self.num_bits
        # )
        # fp_inp = QTensor(
        #     inp.round(), scale=in_scale, zero=in_zero
        # ).dequantize()

        out = self.fp_module(fakeq_inp, *args, **kwargs)

        fakeq_out = self.out_listener(out)

        return fakeq_out

    def forward_quantized(self, inp: QTensor, *args, **kwargs) -> QTensor:

        fp_inp = inp.dequantize()

        fp_out = self.fp_module(fp_inp, *args, **kwargs)

        q_out = self.quantization.quantize_to_qtensor_using_params(
            fp_out,
            scale=self.out_scale_next,
            zero=self.out_zero_next,
            num_bits=self.num_bits
        )

        return q_out

class QReLU6(QuantizableModule):

    def forward_fp(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.relu6(x)

    def forward_qat(self, x: torch.Tensor) -> torch.Tensor:
        min_val = self.__stats__.get("ema_min", x.min().item())
        max_val = self.__stats__.get("ema_max", x.max().item())

        scale, zero = self.quantization.calc_params(
            min_val, max_val, num_bits=self.num_bits
        )
        six = round(6 / scale + zero)
        out = x.clamp(min=zero, max=six)
        return out

    def forward_quantized(self, x: QTensor) -> QTensor:

        scale = self.scale_next
        zero = self.zero_next
        six = round(6 / scale + zero)
        out = x._t.clamp(min=zero, max=six)

        assert round(scale, 5) == round(x.scale, 5), \
                (scale, x.scale)
        assert round(zero, 5) == round(x.zero, 5), \
                (zero, x.zero)

        return QTensor(out, scale=scale, zero=zero)



class ConvBNfoldable(QuantizableModule):
    """
    relu(batchnorm(conv(x))) style module with custom forward pass thats altered during qat_prepare and qat_convert

    via https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html#quantization-aware-training
    as described in https://arxiv.org/abs/1712.05877v1 Sec 3.2


    This module

    Switches the Fig C8 procedure upon call of self.qat_prepare():
    https://bluemountain.eee.hku.hk/papaa2018/PAPAA18-L04-Jac+18.pdf
    Then folds/removes the BN completely when self.fold() is called
    """
    def __init__(
            self,
            in_planes,
            out_planes,
            kernel_size, #=3,
            stride=1,
            groups=1,
            padding=0,
            momentum=0.1,
            relu:Union[int, bool]=6,
            eps=1e-05,
            Convclass=nn.Conv2d,
            **qkwargs
        ):

        super().__init__(**qkwargs)

        if not type(padding) == int:
            # dont know what this is for; its from https://pytorch.org/tutorials/advanced/static_quantization_tutorial.htm
            padding = (kernel_size - 1) // 2

        self.conv = Convclass(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = QBatchNorm2d(out_planes, momentum=momentum)

        self.has_relu = not(type(relu) == bool and relu == False)

        if self.has_relu:
            relu_module = nn.ReLU() if (type(relu)==int and relu==0) else nn.ReLU6()
            self.relu = relu_module

        def debug_fold_backward(module, grad_in, grad_out):
            # sanity check if weights have been updated since last backward pass
            convweight = module.conv.weight.data
            bnweight = module.bn.weight.data
            bnbias = module.bn.bias.data
            if hasattr(module, "last_weights_cache"):
                # are these ever updated?
                if not (convweight == module.last_weights_cache[0]).all():
                    print("conv weight updated!")
                if not (bnweight == module.last_weights_cache[1]).all():
                    print("bn weight updated!")
                if not (bnbias == module.last_weights_cache[2]).all():
                    print("bn bias updated")

            module.last_weights_cache = [convweight]
            module.last_weights_cache += [bnweight]
            module.last_weights_cache += [bnbias]

        # self.register_backward_hook(
        #        debug_fold_backward
        #        )

    def forward_fp(self, x):
        # forward used during fp32 pretraining
        assert not (not self.conv.training and self.bn.track_running_stats)

        x = self.conv(x)
        x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)
        return x

    def forward_quantized(self, x):
        # forward used after conversion, no bn
        x = self.conv(x)
        if self.has_relu:
            x = self.relu(x)
        return x

    def quantize(self):
        super().quantize()
        self.fold()

    def folded_weight(self):
        # C8: w_fold = w * (gamma/sigma)
        folded_weight = (self.conv.weight * (self.bn.weight / torch.sqrt(self.bn.running_var + self.bn.eps)).unsqueeze(1).unsqueeze(1).unsqueeze(1))
        return folded_weight

    def folded_bias(self):
        # C8: bias = beta - gamma * mu / sigma
        folded_bias = (self.bn.bias - ( (self.bn.weight * self.bn.running_mean) / torch.sqrt(self.bn.running_var + self.bn.eps)))
        return folded_bias

    def fold(self):

        folded_weight = self.folded_weight()
        folded_bias = self.folded_bias()

        assert not torch.isnan(folded_weight).any()
        assert not torch.isnan(folded_bias).any()

        self.conv.weight.data = folded_weight
        if self.conv.bias is not None:
            self.conv.bias.data = folded_bias
        else:
            self.conv.bias = nn.Parameter(folded_bias)

        # change function to normal fwd pass again, but wthout bn
        self.forward = self.forward_folded


    def forward_qat(self, x):
        """
        https://bluemountain.eee.hku.hk/papaa2018/PAPAA18-L04-Jac+18.pdf Fig C8 procedure
        """
        assert not (not self.conv.training and self.bn.track_running_stats)

        if self.bn.track_running_stats:
            # update BN running stats
            self.bn(self.conv(x))

        # fold the weights then fake quantize in conv's fwd hook:

        # unterer Teil des training graphs in C8:

        folded_weight = self.folded_weight()
        folded_weight.data = self.conv._fakeQ(folded_weight.data, self.conv._Qwt, self.conv._num_bits_wt, None, None)

        folded_bias = self.folded_bias()
        folded_bias.data = self.conv._fakeQ(folded_bias.data, self.conv._Qwt, self.conv._num_bits_bias, None, None)

        assert not torch.isnan(folded_weight).any()
        assert not torch.isnan(folded_bias).any()

        # DEBUG: try adding bias with functional batch norm instead ..?

        # oberer teil des training graphs in C8:
        x = F.conv2d(
            F.pad(
                x,
                self.conv._reversed_padding_repeated_twice if hasattr(self.conv, "_reversed_padding_repeated_twice") else tuple(x_ for x_ in reversed(self.conv.padding) for _ in range(2)),
            ),
            folded_weight,
            folded_bias,
            self.conv.stride,
            _pair(0),
            self.conv.dilation,
            self.conv.groups,
        )

        if self.has_relu:
            x = self.relu(x)

        return x


def DiscreteHartleyTransform(input):
    # from https://github.com/AlbertZhangHIT/Hartley-spectral-pooling/blob/master/spectralpool.py
    # TODO test as alternative
    fft = torch.rfft(input, 2, normalized=True, onesided=False)
    dht = fft[:, :, :, :, -2] - fft[:, :, :, :, -1]
    return dht

class FFT(nn.Module):
    # move this to ..modules.py

    # TODO implement init and make quantizable FIXME

    # TODO figure out mask

    # figure out whether to orthonormalize (scale by 1/sqrt(n))
    # paper: vandermonde matrix has normalization
    # third party code. no normalization

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):

        # x = fft(fft(x, dim=-1), dim=-2).real
        x = fft2(x)
        x = x.real #  + x.imag
        return x


