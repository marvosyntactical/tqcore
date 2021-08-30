import torch

from .quantized_layer import *
from .quantizable_layer import *
from .quantization_functions import *
import torch.nn as nn
import torch.nn.functional as F

import copy
from typing import Dict

import warnings

# custom layers impl procedure:

# What needs to be implemented for any layer:
# 1. qat_layer_forward_pre_hook (specific fake Q)
# 2. quantized layer function

# record EMA values for all instances of these classes (also wrapper class instances)

# code to prepare QAT training

# pre_hook = call before module.forward
# post_hook = hook = call after module.forward


def _qat_layer_forward_pre_hook(mod:nn.Module, inp:torch.Tensor) -> None:
    """
    mod must be nn.Module with parameters 'weight' and 'bias' and have attributes:
    mod._fakeQ of type FakeQuant
    mod._Qwt of type tinyquant.quantization_functions.Quantization
    mod._num_bits_inp of type int
    """
    # use for linear, conv, batchnorm

    # weights are not clamped to range (hence None, None for min_val, max_val)
    mod.weight.data = mod._fakeQ(
        mod.weight.data, mod._Qwt, mod._num_bits_wt, None, None, handling_qtensors=False
    )

    if mod.bias is not None:
        mod.bias.data = mod._fakeQ(
            mod.bias.data, mod._Qwt, mod._num_bits_bias, None, None, handling_qtensors=False
        )

def qat_prepare(
        module: nn.Module,
        module_number: int = 0,
        parent_number: int = 0,
        quant_input: Quantization = UniformQuantization,
        quant_weight: Quantization = UniformSymmetricQuantization,
        num_bits_input: int = 6,
        num_bits_weight: int = 6,
        num_bits_bias: int = 6,
        inplace: bool = True,
        is_root_module: bool = True,
        _handles: Dict = None,
        _module_types: Dict = None, # convenience to search for next activation during conversion
    ) -> (nn.Module, int):
    """
    prep module for QAT,
    after calling this, train for a few epochs, then convert_model
    """

    if inplace == False and is_root_module:
        # deepcopy root Module but not lower ones
        module = copy.deepcopy(module)

    if _handles is None:
        assert is_root_module
        module.handles = {}
        _handles = module.handles
    if _module_types is None:
        assert is_root_module
        module.module_types = {}
        _module_types = module.module_types

    mod_type = type(module)

    descendants_module_number = module_number + 1

    module._module_number = module_number
    module._parent_number = parent_number

    is_nonquantizable = True in [issubclass(mod_type, layer) for layer in NONQUANT]
    # if is_nonquantizable:
    #     module.qat_prepare()
    #     # do not recurse to children
    #     return module, descendants_module_number

    named_children = dict(module.named_children())

    if is_nonquantizable:
        # ignore module wrapped by nonquant module wrapper, but dont ignore its listeners
        del named_children["fp_module"]

    # increment DFS counter by 1 for current module before diving down further
    for name, layer in named_children.items():

        # ===== DFS down module graph ========
        _, descendants_module_number = qat_prepare(
            layer,
            descendants_module_number, # numbering is DFS preorder
            module_number,
            quant_input=quant_input,
            quant_weight=quant_weight,
            num_bits_input=num_bits_input,
            num_bits_weight=num_bits_weight,
            num_bits_bias=num_bits_bias,
            inplace=True, # only deepcopy once, at top level
            is_root_module=False,
            _handles=_handles,
            _module_types=_module_types
        )

    # ---------- PREPARE DIFFERENT MODULES CASE BY CASE --------- #

    _handles[module_number] = dict()
    _module_types[module_number] = type(module)
    param_names = [name for name, _ in module.named_parameters()]

    if isinstance(module, QuantizableModule):
        # most custom modules in .quantizable_layer; _QBatchNorm in .batchnorm
        module.qat_prepare()

    elif isinstance(module, nn.Linear) or \
           isinstance(module, nn.Conv2d) or \
           isinstance(module, nn.modules.batchnorm._BatchNorm):

        assert not set(param_names) - {"weight", "bias"}, param_names

        module._Qwt = quant_weight()
        module._num_bits_wt = num_bits_weight
        module._num_bits_bias = num_bits_bias
        module._fakeQ = FakeQuant.apply_wrapper

        # fake quantize weights, bias
        pre_hook_handle = module.register_forward_pre_hook(_qat_layer_forward_pre_hook)

        _handles[module_number]["qat_pre"] = pre_hook_handle

        # FIXME current converted fwd implementation uses Qinp for input quantization ...
        # but module records no Qinp stats during QAT... TODO
        module._Qinp = quant_input()
        module._num_bits_inp = num_bits_input

    elif isinstance(module, nn.MaxPool2d):
        module._Qinp = quant_input()
        module._Qwt = quant_weight()
        module._num_bits_wt = num_bits_weight
        module._num_bits_bias = num_bits_bias
        module._num_bits_inp = num_bits_input

    elif param_names and not len(module._modules):
        # implement other layer types if this is raised, good luck
        warnings.warn(f"Dont know how to prepare parameterized leaf module '{module_number}', an instance of {type(module)}")
    else:
        is_leaf = descendants_module_number == module_number + 1
        if is_leaf:
            print(f"Don't know how to prepare leaf module '{module_number}', an instance of {type(module)}")


    # ---------- END PREPARE DIFFERENT MODULES CASE BY CASE ---------

    return module, descendants_module_number


