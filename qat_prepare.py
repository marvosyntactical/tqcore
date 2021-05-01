import torch

from .quantized_layer import *
from .quantizable_layer import *
from .quantization_ema_stats import *
from .quantization_functions import *
import torch.nn as nn
import torch.nn.functional as F

import copy
from typing import Dict

# custom layers impl procedure:

# What needs to be implemented for any layer:
# 1. qat_layer_forward_pre_hook (specific fake Q)
# 2. quantized layer function

# record EMA values for all instances of these classes (also wrapper class instances)
global ACTIVS
ACTIVS = [nn.ReLU, nn.ReLU6, nn.Identity] #, QuantizableResConnection]

# code to prepare QAT training

# pre_hook = call before module.forward
# post_hook = hook = call after module.forward

def _factory_track_hook_ema_module_forward(stats, key):
    def _track_hook_ema_module_forward(mod, inp, res):
        training = mod.training if not hasattr(mod,"track_running_stats") else mod.track_running_stats
        if training:
            update_ema_stats(res, stats, key)
    return _track_hook_ema_module_forward

def _qat_layer_forward_pre_hook(mod:nn.Module, inp:torch.Tensor) -> None:
    """
    mod must be nn.Module with parameters 'weight' and 'bias' and have attributes:
    mod._fakeQ of type FakeQuant
    mod._Qwt of type tinyquant.quantization_functions.Quantization
    mod._num_bits_inp of type int
    """

    # weights are not clamped to range (hence None, None for min_val, max_val)
    mod.weight.data = mod._fakeQ(mod.weight.data, mod._Qwt, mod._num_bits_wt, None, None)

    if mod.bias is not None:
        mod.bias.data = mod._fakeQ(mod.bias.data, mod._Qwt, mod._num_bits_bias, None, None)


def _factory_qat_activation_forward_hook(stats: Dict) -> torch.Tensor:
    def _qat_activation_forward_hook(mod: nn.Module, inp: torch.Tensor, res:torch.Tensor) -> torch.Tensor:
        """
        mod must be nn.Module activation (only ReLU or Identity for now)
        mod._fakeQ of type FakeQuant
        mod._Qinp of type tinyquant.quantization_functions.Quantization
        mod._num_bits_inp of type int
        """
        module_number = mod._module_number

        ema_min, ema_max = \
            stats[module_number]["ema_min"], \
            stats[module_number]["ema_max"]

        # fake quantize activations
        res_fq = mod._fakeQ(
            res,
            mod._Qinp,
            mod._num_bits_inp,
            ema_min,
            ema_max
        )

        return res_fq
    return _qat_activation_forward_hook


def prep_module_qat(
        module: nn.Module,
        module_number: int = 0,
        parent_number: int = 0,
        quant_input: Quantization = UniformQuantization(),
        quant_weight: Quantization = UniformSymmetricQuantization(),
        num_bits_input: int = 6,
        num_bits_weight: int = 6,
        num_bits_bias: int = 6,
        inplace: bool = True,
        _stats: Dict = None,
        _handles: Dict = None,
        _module_types: Dict = None, # convenience to search for next activation during conversion
    ) -> (nn.Module, Dict):
    """
    prep module for qat,
    return dict of forward hook handles to call handle.remove() on during conversion.

    after calling this, train for a few epochs, then convert_model
    """

    nested_suffix = "_" # to disambiguate modules that have the same name as father, father gets renamed to "fathername_"; use this in conversion... FIXME BUG

    is_root_module = _stats is None

    if inplace == False and is_root_module:
        # deepcopy root Module but not lower ones
        module = copy.deepcopy(module)

    # TODO put all these dicts into one
    # only outer module gets dict attributes, descendants point to and modify it in place
    if _stats is None:
        module.stats = {}
        _stats = module.stats
    if _handles is None:
        module.handles = {}
        _handles = module.handles
    if _module_types is None:
        module.module_types = {}
        _module_types = module.module_types

    descendants_module_number = module_number + 1 # increment DFS counter by 1 for current module before diving down further

    for name, layer in module.named_children():

        # ===== DFS down module graph ========
        _, descendants_module_number = prep_module_qat(
            layer,
            descendants_module_number, # numbering is DFS preorder
            module_number,
            quant_input=quant_input,
            quant_weight=quant_weight,
            num_bits_input=num_bits_input,
            num_bits_weight=num_bits_weight,
            num_bits_bias=num_bits_bias,
            inplace=True, # only deepcopy once, at top level
            _stats=_stats,
            _handles=_handles,
            _module_types=_module_types
        )

    module._module_number = module_number
    module._parent_number = parent_number
    _handles[module_number] = dict()
    _module_types[module_number] = type(module)

    if True in [isinstance(module, activ) for activ in ACTIVS]:
        ema_handle = module.register_forward_hook(
            _factory_track_hook_ema_module_forward(_stats, module_number)
        )
        _handles[module_number]["ema_handle"] = ema_handle

    param_names = [name for name, _ in module.named_parameters()]

    # ---------- PREPARE DIFFERENT MODULES CASE BY CASE --------- #

    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):

        assert not len(set(param_names) - {"weight", "bias"}), param_names

        module._Qwt = quant_weight
        module._num_bits_wt = num_bits_weight
        module._num_bits_bias = num_bits_bias
        module._fakeQ = FakeQuant.apply

        # update fwd hooks
        pre_hook_handle = module.register_forward_pre_hook(_qat_layer_forward_pre_hook) # fake quantize weight

        _handles[module_number]["qat_pre"] = pre_hook_handle

        # FIXME current converted fwd implementation uses Qinp for input quantization ...
        # but module records no Qinp stats during QAT... TODO
        module._Qinp = quant_input
        module._num_bits_inp = num_bits_input

    elif isinstance(module, ConvBNfoldable):

        module.qat_prepare()

    elif isinstance(module, ConvBNnofold):
        # FIXME regel das auf bessere Weise:
        # im Falle von nofold: will aktivierungswerte von vor QAT

        bnpos = str(1)
        module._modules[bnpos]._Qwt = quant_weight
        module._modules[bnpos]._num_bits_wt = num_bits_weight
        module._modules[bnpos]._num_bits_bias = num_bits_bias
        module._modules[bnpos]._fakeQ = FakeQuant.apply
        module._modules[bnpos]._Qinp = quant_input
        module._modules[bnpos]._num_bits_inp = num_bits_input

        # update weights
        pre_hook_handle = module._modules[bnpos].register_forward_pre_hook(_qat_layer_forward_pre_hook) # fake quantize weight
        _handles[module_number]["qat_pre"] = pre_hook_handle

    elif isinstance(module, nn.MaxPool2d):
        module._Qinp = quant_input
        module._Qwt = quant_weight
        module._num_bits_wt = num_bits_weight
        module._num_bits_bias = num_bits_bias
        module._num_bits_inp = num_bits_input

    elif isinstance(module, nn.modules.batchnorm._BatchNorm):
        pass

    elif isinstance(module, QuantizableResConnection):
        # FIXME not actually used. see above TODO in conv2d/linear case
        module._Qwt = quant_weight
        module._num_bits_wt = num_bits_weight

        # actually used hook handles:
        module._Qinp = quant_input
        module._num_bits_inp = num_bits_input
        module._fakeQ = FakeQuant.apply

    elif True in [isinstance(module, activ) for activ in ACTIVS]:

        module._Qinp = quant_input
        module._num_bits_inp = num_bits_input
        module._fakeQ = FakeQuant.apply

        post_hook_handle = module.register_forward_hook(
            _factory_qat_activation_forward_hook(_stats)
        ) # fake quantize activations
        _handles[module_number]["qat_post"] = post_hook_handle

    elif param_names and not len(module._modules):
        # implement other layer types if this is raised, good luck
        raise NotImplementedError(f"Dont know how to prepare parameterized leaf module '{module_number}', an instance of {type(module)}")
    else:
        is_leaf = descendants_module_number == module_number + 1
        if is_leaf:
            print(f"Don't know how to prepare leaf module '{module_number}', an instance of {type(module)}")

    if is_root_module:
        module._Qinp = quant_input
        module._Qwt = quant_weight
        module._num_bits_wt = num_bits_weight
        module._num_bits_bias = num_bits_bias
        module._num_bits_inp = num_bits_input

    # ---------- END PREPARE DIFFERENT MODULES CASE BY CASE ---------

    return module, descendants_module_number
