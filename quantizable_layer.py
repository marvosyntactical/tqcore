import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from .quantization_functions import QTensor, Quantization
from .batchnorm import *

from torch.nn.modules.utils import _pair

import math
import copy
from typing import Optional, Union
from functools import partial


# this module contains quantizable versions of nn.Modules

# TODO import stuff from quantized_layer

MOMENTUM = .01
__DEBUG__ = False

# helper fns
printdbg = lambda *expr: print(*expr) if __DEBUG__ else None
tnsr_stats = lambda t, qinp: (round(t.min().item(), 3), round(t.max().item(), 3),qinp.calc_zero_point(t.min().item(), t.max().item(), 8))
qparams_stats = lambda qparams_dict: tuple(map(lambda fl: round(fl, 3),qparams_dict.values()))
is_integer = lambda t: (t.round()==t).all()


class QuantizableModel(nn.Module):
    # TODO  vvvvvvvvvvv
    def __init__(self, model:nn.Module, *args, **kwargs):
        """
        Model must be initialized instance of nn.Module
        after calling qat_prepare on it
        and after doing QAT.
        call this during conversion to fully quantized model
        """
        super().__init__()

        self.quant_stub = Quant()
        self.model = model

        # retrieve min and max vales for input of module itself
        # first_activation_number = search_for_next_activ(_module_types, 0)


    def forward(self, *args, **kwargs) -> Tensor:
        """
        Replace every Tensor in args, kwargs by QTensor.
        Pass everything to model.
        Replace every output QTensor with Tensor.
        """
        for i, inp in enumerate(args):
            if isinstance(inp, Tensor)\
                    and torch.is_floating_point(inp)\
                    and not isinstance(inp, QTensor):
                args[i] = QTensor(inp)
                args[i] = self._Qinp.quantize_to_qtensor(
                    inp,
                    self.min_val,
                    self.max_val,
                    num_bits=self.model._num_bits_inp
                )
        for key, inp in enumerate(kwargs):
            if isinstance(inp, Tensor)\
                    and torch.is_floating_point(inp)\
                    and not isinstance(inp, QTensor):
                kwargs[key] = QTensor(inp)
                kwargs[key] = self._Qinp.quantize_to_qtensor(
                    inp,
                    self.min_val,
                    self.max_val,
                    num_bits=self.model._num_bits_inp
                )
        # forward
        outputs = self.model(*args, **kwargs)

        if not isinstance(outputs, tuple):
            assert isinstance(outputs, Tensor), type(outputs)
            outputs = (outputs,)

        for i, out in enumerate(outputs):
            if isinstance(out, QTensor):
                outputs[i] = self._Qinp.dequantize(out)
        return outputs

class QuantizableModule(nn.Module):
    """
    During fp training, this module acts as identity.
    It also has a Quantization Aware Training (QAT) stage, and quantized stage,
    which submodules must both implement
    """

    def __init__(self):
        super().__init__()
        self.forward = self.forward_fp

    def forward_fp(self, x: Tensor) -> Tensor:
        return x

    def forward_qat(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def forward_quantized(self, x: QTensor) -> QTensor:
        raise NotImplementedError

    def qat_prepare():
        self.forward = self.forward_qat

    def quantize(self):
        self.forward = self.forward_quantized

class QListener(QuantizableModule):
    """
    During qat, this module records min, max values of torch.Tensor s passing through.
    (no other module records stats)
    Accepts an iterable of modules for each of which the QListener sets the module.__stats__ attribute
    """
    def __init__(self, *modules, num_bits: int = 8, ema_decay: float = .9999, nudge_zero: bool = False):
        super().__init__()

        self.ema_decay = ema_decay
        self.stats = {}

        for module in modules:
            module.__stats__ = self.stats

        self.num_bits = num_bits
        self.quantization = UniformQuantization(nudge_zero=nudge_zero)
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
            stats["ema_min"],
            stats["ema_max"]
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

        max_val = torch.max(x).item()
        min_val= torch.min(x).item()
        # assert max_val != min_val, (max_val, (x==max_val).all())

        # if layer not yet in dictionary: create EMA for layer
        if key not in self.stats:
            self.stats = {"max": max_val, "min": min_val}
        else:
            curr_max = self.stats["max"]
            curr_min = self.stats["min"]
            self.stats['max'] = max(max_val, curr_max) if curr_max is not None else max_val
            self.stats['min'] = max(min_val, curr_min) if curr_min is not None else min_val


        if 'ema_min' in self.stats:
            # self.stats['ema_min'] = (1.-self.ema_decay) * min_val + self.ema_decay * self.stats['ema_min']
            self.stats['ema_min'] -=  (1 - self.ema_decay) * (self.stats['ema_min'] - min_val)

        else:
            self.stats['ema_min'] = min_val

        if 'ema_max' in self.stats:
            # self.stats['ema_max'] = (1.-self.ema_decay) * max_val + self.ema_decay * self.stats['ema_max']
            self.stats['ema_max'] -= (1 - self.ema_decay) * (self.stats['ema_max'] - max_val)
        else:
            self.stats['ema_max'] = max_val

class QScale(QuantizableModule):

    pass

class QMatMul(QuantizableModule):
    pass

class QAdd(QuantizableModule):
    pass

class QSoftmax(QuantizableModule):

    def __init__(self, num_bits: int):
        super().__init__()
        self.set_exp_lkp(num_bits)

    def set_exp_lkp(self, num_bits:int):
        self.LUT = torch.exp(torch.arange(2.** num_bits)).round()



class ConvBNfoldable(ConvBNContainer, QuantizableModule):
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
            momentum=MOMENTUM,
            relu:Union[int, bool]=6,
            eps=1e-05,
            Convclass=nn.Conv2d,
            BNclass=BatchNorm2dWrap,
        ):

        super(ConvBNfoldable, self).__init__()

        if not type(padding) == int:
            # dont know what this is for; its from https://pytorch.org/tutorials/advanced/static_quantization_tutorial.htm
            padding = (kernel_size - 1) // 2

        self.conv = Convclass(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = BNclass(out_planes, momentum=momentum)

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

    def folded_weight(self):
        # C8: w_fold = w * (gamma/sigma)
        folded_weight = (self.conv.weight * (self.bn.weight / torch.sqrt(self.bn.running_var + self.bn.eps)).unsqueeze(1).unsqueeze(1).unsqueeze(1))
        return folded_weight

    def folded_bias(self):
        # C8: bias = beta - gamma * mu / sigma
        folded_bias = (self.bn.bias - ( (self.bn.weight * self.bn.running_mean) / torch.sqrt(self.bn.running_var + self.bn.eps)))
        return folded_bias

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

    def fold(self):
        # called during qat_convert

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


