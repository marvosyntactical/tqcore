import torch
from torch import Tensor
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

# this module contains quantizable versions of basic nn.Modules, as well as some helper modules

class QuantizableModule(nn.Module):
    """
    During fp training, this module acts as Identity.
    It also has a Quantization Aware Training (QAT) stage, and quantized stage,
    which submodules should implement (otherwise this module remains an Identity)
    """

    def __init__(
            self,
            quantization: Quantization = UniformQuantization,
            weight_quantization: Quantization = UniformSymmetricQuantization,
            num_bits: int = 8,
            num_bits_weight: int = 8,
            nudge_zero: bool = False,
            **qkwargs,
        ):
        super().__init__()

        self.num_bits = num_bits
        self.quantization = quantization(nudge_zero=nudge_zero)
        self.num_bits_weight = num_bits_weight
        self.weight_quantization = weight_quantization(nudge_zero=nudge_zero)

        self.forward = self.forward_fp

    def forward_fp(self, x: Tensor) -> Tensor:
        return x

    def forward_qat(self, x: Tensor) -> Tensor:
        raise NotImplementedError()

    def forward_quantized(self, x: QTensor) -> QTensor:
        raise NotImplementedError()

    def qat_prepare(self):
        self.forward = self.forward_qat

    def quantize(self):
        self.forward = self.forward_quantized

        stats_dict_names = [attr[:attr.find("__stats__")] for
            attr in dir(self) if attr.endswith("__stats__")
        ]
        assert stats_dict_names, f"some <name>+__stats__ of {type(self)} needs to be set by a QListener module"
        for name in stats_dict_names:
            stats = getattr(self, name+"__stats__")

            prefix = name + "_" if name else ""
            # prohpylactically set qparams for all QuantizableModules

            scale_next, zero_next = self.quantization.calc_params(stats["ema_min"], stats["ema_max"], num_bits=self.num_bits)

            setattr(self, prefix+"scale_next", scale_next)
            setattr(self, prefix+"zero_next", zero_next)


class Quant(QuantizableModule):
    """
    Quantizes incoming torch.Tensors into tqcore.QTensors if necessary.
    Functionality Analogous to torch.quantization.QuantStub
    """
    def forward_quantized(self, *args, **kwargs):
        for i, inp in enumerate(args):
            if isinstance(inp, Tensor)\
                    and torch.is_floating_point(inp)\
                    and not isinstance(inp, QTensor):
                args[i] = self.quantization.quantize_to_qtensor(
                    inp,
                    self.min_val,
                    self.max_val,
                    num_bits=self.num_bits
                )

        for key, inp in enumerate(kwargs):
            if isinstance(inp, Tensor)\
                    and torch.is_floating_point(inp)\
                    and not isinstance(inp, QTensor):
                kwargs[key] = self._Qinp.quantize_to_qtensor(
                    inp,
                    self.min_val,
                    self.max_val,
                    num_bits=self.num_bits
                )
        return args, kwargs

class DeQuant(QuantizableModule):
    """
    Dequantizes incoming torch.Tensors into tqcore.QTensors if necessary.
    Functionality Analogous to torch.quantization.QuantStub
    """
    def forward_quantized(self, outputs):
        if not isinstance(outputs, tuple):
            assert isinstance(outputs, Tensor), type(outputs)
            outputs = (outputs,)

        outputs = list(outputs)

        for i, out in enumerate(outputs):
            if isinstance(out, QTensor):
                outputs[i] = out.dequantize()
        return outputs

class QListener(QuantizableModule):
    """
    During qat, this module records min, max values of torch.Tensor s passing through.
    (no other module records stats)
    Accepts an iterable of modules for each of which the QListener sets the module.scale_next attribute and so on
    """
    def __init__(self, *modules: nn.Module, name = None, function = None, ema_decay: float = .9999, nudge_zero: bool = False, **qkwargs):
        super().__init__(**qkwargs)

        self.function = function # optionally apply function before collecting stats (for softmax)
        self.name = "" if name is None else str(name) # set attribute name_scale_next and so on
        self.ema_decay = ema_decay

        self.__stats__ = {}
        for module in modules:
            setattr(module, self.name+"__stats__", self.__stats__)

        self.fake_quantize = FakeQuant.apply

    def forward_qat(self, x: Tensor):
        """
        Collect stats AND fakequantize
        """
        self._update_ema_stats(x)
        scale, zero = self.quantization.calc_params(self.stats["ema_min"], self.stats["ema_max"], num_bits=self.num_bits)

        x = self.fake_quantize(
            x,
            self.quantization,
            self.num_bits,
            self.__stats__["ema_min"],
            __stats__["ema_max"]
        )
        return x

    def forward_quantized(self, x: QTensor) -> QTensor:
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
            x = self.function(x)

        max_val = torch.max(x).item()
        min_val = torch.min(x).item()
        # assert max_val != min_val, (max_val, (x==max_val).all())

        # if layer not yet in dictionary: create EMA for layer
        if key not in self.__stats__:
            self.__stats__ = {"max": max_val, "min": min_val}
        else:
            curr_max = self.__stats__["max"]
            curr_min = self.__stats__["min"]
            self.__stats__['max'] = max(max_val, curr_max) if curr_max is not None else max_val
            self.__stats__['min'] = max(min_val, curr_min) if curr_min is not None else min_val


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


def _qmul(
        a: QTensor, b: QTensor,
        scale_next, zero_next, torchOp,
        quantization, weight_quantization,
        num_bits, num_bits_weight):
    # helper func for mul and matmul
    # TODO future:
    # replace this and QAdd.forward_quantized by gemmlowp (possibly <) 8 bit kernel

    ab = [a,b]
    for i, t in enumerate(ab):
        if not isinstance(t, QTensor):
            if isinstance(t, torch.Tensor):
                if isinstance(t, torch.Parameter):
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
                t = QTensor(torch.as_tensor(t), scale=1., zero=0.)
        ab[i] = t
    a, b  = ab

    a_zeroed = a._t - a.zero
    b_zeroed = b._t - b.zero

    r: torch.Tensor = torchOp(a_zeroed, b_zeroed)

    multiplier = (a.scale * b.scale) / scale_next
    # scale result tensor back to given bit width, saturate to uint if unsigned is used:
    r = r * multiplier + zero_next
    # round and clamp
    r = quantization.tensor_clamp(r, num_bits=num_bits)

    assert is_integer(r), r

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
        return _qmul(a, b, self.scale_next, self.zero_next, torch.mul,
                self.quantization, self.weight_quantization,
                self.num_bits, self.num_bits_weight)


class QMatMul(QuantizableModule):
    def __init__(self, *args, **qkwargs):
        super().__init__(*args, **qkwargs)

    def forward_fp(self, a, b):
        return a @ b

    def forward_qat(self, a, b):
        return a @ b

    def forward_quantized(self, a: QTensor, b:QTensor) -> QTensor:
        return _qmul(
                a, b, self.scale_next, self.zero_next, torch.matmul,
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
            zero=.5*self.next_zero,
            num_bits=self.num_bits-1
        )
        b_requantized = self.quantization.quantize_to_qtensor_using_params(
            b.dequantize(),
            scale=1/(.5*self.scale_next),
            zero=.5*self.next_zero,
            num_bits=self.num_bits-1
        )

        r = a_requantized + b_requantized

        assert is_integer(r), r

        return r

class QMask(QuantizableModule):
    def __init__(self, float_neg_val: float=float("-inf"), **qkwargs):
        super().__init__(**qkwargs)
        self.float_neg_val = float(float_neg_val)

    def forward_fp(self, scores, mask):
        scores = scores.masked_fill(mask==torch.as_tensor(False), float('-inf'))
        return scores

    def forward_qat(self, scores, mask):
        scores = scores.masked_fill(mask==torch.as_tensor(False), float('-inf'))
        return scores

    def forward_quantized(self, scores, mask):
        scores = scores.masked_fill(mask==torch.as_tensor(False), self.zero_next)
        return scores

class QSoftmax(QuantizableModule):

    def __init__(self, dim=-1, **qkwargs):
        super().__init__(**qkwargs)

        self._set_exp_lkp(qkwargs["num_bits"])
        self.dim = int(dim)

        # this records EMA stats of exponential, and has no fake quant effect
        self.exp_listener = QListener(self, name="exp", function=torch.exp, **qkwargs)

        # this ALSO records stats of normed output (which will be between 0 and 1 anyway),
        # but is used mainly because of fake quantizing
        self.norm_listener = QListener(self, name="normed", **qkwargs)

    def _set_exp_lkp(self, num_bits:int):
        # this range is tailored to uniformquantization
        self.LUT = torch.exp(torch.arange(2.** num_bits)).round()

    def forward_fp(self, inp: torch.Tensor) -> torch.Tensor:
        return inp.softmax(dim=self.dim)

    def forward_qat(self, inp: torch.Tensor) -> torch.Tensor:
        _ = self.exp_listener(inp)
        out = inp.softmax(dim=self.dim)
        out = self.norm_listener(out)
        return out

    def forward_quantized(self, inp: QTensor) -> QTensor:

        exponentiated: torch.Tensor = self.LUT[inp._t]

        numerator = QTensor(exponentiated, scale=self.exp_scale_next, zero=self.exp_zero_next)
        denominator = QTensor(1/exponentiated.sum(dim=self.dim).unsqueeze(-1),
            scale=1/self.exp_scale_next, zero=1/self.exp_zero_next)

        r = _qmul(
            numerator, denominator,
            self.normed_scale_next, self.normed_zero_next,
            torch.mul,
            self.quantization, self.weight_quantization,
            self.num_bits, self.num_bits_weight
        )

        return r

class QReLU6(QuantizableModule):

    def forward_fp(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.relu6(x)

    def forward_qat(self, x: torch.Tensor) -> torch.Tensor:
        scale, zero = self.quantization.calc_params(
            self.__stats__["ema_min"], self.__stats__["ema_max"], num_bits=self.num_bits
        )
        six = round(6 / scale_next + zero)
        out = x.clamp(min=zero, max=six)
        return r

    def forward_quantized(self, x: QTensor) -> QTensor:

        scale = self.scale_next
        zero = self.zero_next
        six = round(6 / scale_next + zero)
        out = x.clamp(min=zero, max=six)

        assert round(scale, 5) == round(x.scale, 5), \
                (scale, x.scale)
        assert round(zero, 5) == round(x.zero, 5), \
                (zero, x.zero)

        return out



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


