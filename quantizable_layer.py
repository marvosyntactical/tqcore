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

# this module contains quantizable versions of nn.Modules

# TODO import stuff from quantized_layer

MOMENTUM = .01
__DEBUG__ = False

# helper fns
printdbg = lambda *expr: print(*expr) if __DEBUG__ else None
tnsr_stats = lambda t, qinp: (round(t.min().item(), 3), round(t.max().item(), 3),qinp.calc_zero_point(t.min().item(), t.max().item(), 8))
qparams_stats = lambda qparams_dict: tuple(map(lambda fl: round(fl, 3),qparams_dict.values()))
is_integer = lambda t: (t.round()==t).all()


class QuantizableResConnection(nn.Module):
    """
    Residual Connection with methods that are quantized during qat_convert, with new implementations in quantized_layer

    ## How your model.init and model.forward must be changed for this module
    ### Before (your normal nn.Module)

    def __init__(self, *args):
        self.first_child = ...
        ...
        self.before_skip = ...
        self.skipping_child = ...
        self.skipped_child = ...
        self.next_activ = ...
        ...
    def forward(self, x)
        x = self.first_child(x)
        ...
        x = self.before_skip(x)
        x = self.skipping_child(x) + self.skipped_child(x)
        x = self.next_activ(x)

    ### After (your nn.Module fitted for QuantizeableResConnection)

    def __init__(self, *args):
        self.first_child = ...
        ...
        self.before_skip = ...
        self.skipping_child = ...
        self.skipped_child = ...
        self.residual = QuantizableResConnection() # must be right before activ
        self.next_activ = ... # insert an nn.Identity() if no activ was here before

    def forward(self, x)
        x = self.first_child(x)
        ...
        x = self.before_skip(x)

        self.residual.cache()
        skip = self.skipping_child(x) # this "branch" needs to have a module that updates self.__qparams__
        self.residual.reset()

        ... # there may be more modules in here

        x = self.residual.add(
            skip, # "older" version of processed batch
            self.skipped_child(x) # "newer" version of processed batch
        )
        x = self.next_activ(x)

    ### Reasoning for this implementation

    #### What has to be done for residuals for correct quantization:

    1. Adjust scale of one of the tensors to the other
    2. Add them
    3. Requantize to the range [0, 2**bits]

    #### Why is this problematic/ How to solve this?

    * The rest of my implementation only stores the current qparams of the tensor in a buffer of the model that every descendant also points to
    * (this is practical for a sequential control flow because the (only?) alternative implementation would store the qparams in a wrapper class around the tensor; but I decided to avoid touching the Tensor itself to not have to get involved with many special cases such as torch.flatten or torch.mean (in hindsight this might still have been much easier))

    * So since I must adjust the scale of the first Tensor in 1., I need to:
    - cache() the qparams before the quantized "skipping_child" updates the global qparams, and
    - reset() the global qparams to the cached ones afterwards so the next part of the control flow (usually the skipped_child) uses the correct qparams before the next activation

    Afterwards, we
    - add() the older version of the batch (first arg) to the newer version (second arg)

    """

    def __init__(self, *args, **kwargs):
        super(QuantizableResConnection, self).__init__(*args, **kwargs)

        # during quantized inference, this module handles
        # qparams for the more recent of the two tensors
        # internally:

        self.is_quantized = False
        self.qparams_newer_tensor = dict()

    def cache(self):
        # gets replaced in qat_convert
        pass

    def reset(self):
        # gets replaced in qat_convert
        pass

    def rescale(self, x):
        # gets replaced in qat_convert
        return x

    def identity(self, new):
        # does nothing to newer tensor until replaced in qat_convert
        return new

    def forward(self, old):
        # this gets changed to a rescaling to next activ range during model conversion
        return old

    def add_impl(self, older, newer):
        """Adds old input batch from earlier in model forward control flow to more recent activations"""

        newfwd = self.identity(newer) # does nothing until quantized, then saves qparams
        oldfwd = self(older)

        sum = oldfwd + newfwd

        if self.is_quantized:
            assert is_integer(sum)

        return sum

    def add(self, older, newer):
        """Adds old input batch from earlier in model forward control flow to newer batch"""
        # this gets replaced in qat_convert
        return self.add_impl(older, newer)

def _convbn_folding_simulation_forward_pre_hook(mod, inp):
    """
    updates batch normalization parameters before actual forward pass of ConvBN like in the lower half of
    the Fig C7 procedure https://bluemountain.eee.hku.hk/papaa2018/PAPAA18-L04-Jac+18.pdf
    """
    # not actually processing input to pass it along network graph, only to update mod.bn stats

    if isinstance(inp, tuple):
        inp = inp[0]
        assert isinstance(inp, torch.Tensor), type(inp)

    # taking caution to call bn.forward to set bn params, but not conv.forward to not call conv pre hook when this is used during QAT!
    inp_after_conv = F.conv2d(
        F.pad(
            inp,
            mod.conv._reversed_padding_repeated_twice if hasattr(mod.conv, "_reversed_padding_repeated_twice") else tuple(x for x in reversed(mod.conv.padding) for _ in range(2)),
        ),
        mod.conv.weight,
        mod.conv.bias,
        mod.conv.stride,
        _pair(0),
        mod.conv.dilation,
        mod.conv.groups,
    )
    # track running stats:
    inp_after_bn = mod.bn(inp_after_conv)


class ConvBNContainer(nn.Module):
    pass



class ConvBNfoldable(ConvBNContainer):
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

    def forward(self, x):
        # forward used during fp32 pretraining
        assert not (not self.conv.training and self.bn.track_running_stats)

        x = self.conv(x)
        x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)
        return x

    def forward_folded(self, x):
        # forward used after conversion, no bn
        x = self.conv(x)
        if self.has_relu:
            x = self.relu(x)
        return x

    def qat_prepare(self):

        self.forward_old = self.forward
        self.forward = self.forward_fold

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


    def forward_fold(self, x):
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




class ConvBNnofold(ConvBNContainer, nn.Sequential):
    """
    via https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html#quantization-aware-training
    """
    def __init__(
            self,
            in_planes,
            out_planes,
            kernel_size, #=3
            stride=1,
            groups=1,
            padding=0,
            momentum=MOMENTUM,
            relu:Union[int, bool]=6,
            eps=1e-05,
            Convclass=nn.Conv2d,
            BNclass=BatchNorm2dWrap,
        ):

        convargs = {
            "in_channels": in_planes,
            "out_channels":out_planes,
            "kernel_size":kernel_size,
            "stride":stride,
            "padding":padding,
            "groups":groups,
        }

        if type(padding) == str:
            # FIXME
            # dont know what this calculation is for; see https://pytorch.org/tutorials/advanced/static_quantization_tutorial.htm
            convargs["padding"] = (kernel_size - 1) // 2
        elif type(padding) == None:
            del convargs["padding"]

        super(ConvBNnofold, self).__init__(
            Convclass(bias=False, **convargs),
            BNclass(out_planes, momentum=momentum, eps=eps)
        )
        if relu == True or type(relu)==int:
            relu_module = nn.ReLU() if (type(relu)==int and relu==0) else nn.ReLU6()
            # activ = nn.ReLU6() if relu else nn.Identity()
            self._modules.update({str(len(self)): relu_module})

        def debug_fold_backward(module, grad_in, grad_out):
            # sanity check if weights have been updated since last backward pass
            convweight = module._modules["0"].weight.data
            bnpos = str(1)
            bnweight = module._modules[bnpos].weight.data
            bnbias = module._modules[bnpos].bias.data
            if hasattr(module, "last_weights_cache"):
                # are these ever updated?
                if not (convweight == module.last_weights_cache[0]).all():
                    print("conv weight updated!!")
                if not (bnweight == module.last_weights_cache[1]).all():
                    print("bn weight updated!!")
                if not (bnbias == module.last_weights_cache[2]).all():
                    print("bn bias updated!!")

            module.last_weights_cache = [convweight]
            module.last_weights_cache += [bnweight]
            module.last_weights_cache += [bnbias]

        # self.register_backward_hook(
        #        debug_fold_backward
        #        )


    def train(self, mode:bool):
        mode = bool(mode)

        # do NOT inform submodules (specifically batch norm) whether we are training
        # super(ConvBNfoldable, self).train(mode=mode)
        # because batchnorm can apparently not handle eval mode
        # (https://github.com/pytorch/pytorch/issues/4741)
        # instead, set following flag and handle it in tinyquant.batchnorm.BatchNormWrap:

        bnpos = str(1) # TODO

        if mode:
            self._modules[bnpos].track_running_stats = True
        else:
            self._modules[bnpos].track_running_stats = False

        self.training = mode
        self._modules[str(0)].training = mode # conv
        if len(self) == 3:
            self._modules[str(2)].training = mode # relu
        # do not recurse to bn
        return self

# unused atm:
# ConvBNFolder used to be here, see history of my repo and also where I took it from:
# from https://github.com/nathanhubens/fasterai/blob/master/fasterai/bn_folder.py
