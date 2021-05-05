from copy import deepcopy
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

TRANSFORMER = True

if TRANSFORMER:
    from .transformer_layers import *

from .quantizable_layer import *

from .quantized_layer import _factory_convert_layer_forward_impl, OPS

from .quantization_functions import *

# using lisa kuhn's code in .tinyquant/*
# to prepare a module for QAT and convert it (like in the pytorch QAT API)

# procedure is analogous to the pytorch QAT API
# (https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html#quantization-aware-training)
# 1. pretrain float model
# 2. call prep_model_qat
# 3. train model some more
# 4. call convert module

FIRST_LAYERS = [nn.Conv2d, nn.Linear]
LAST_LAYERS = [nn.Conv2d, nn.Linear]

def find_first_and_last_layer(module_types_dict) -> Tuple[int]:

    first_idx=0
    found_first = False
    while True:
        mod_type = module_types_dict[first_idx]
        is_layer = True in [issubclass(mod_type, layer) for layer in FIRST_LAYERS]
        if is_layer:
            found_first = True
            break
        first_idx += 1
    assert found_first, "did not find any implemented LAYER to pick as first layer ..."

    last_idx = max(module_types_dict.keys())
    found_last = False
    while True:
        mod_type = module_types_dict[first_idx]
        is_layer = True in [issubclass(mod_type, layer) for layer in LAST_LAYERS]
        if is_layer:
            found_last = True
            break
        last_idx -= 1
    assert found_last, "did not find any implemented LAYER to pick as last layer ..."
    return (first_idx, last_idx)

def quantize_model(
        model: nn.Module,
        **kwargs):

    model = convert_module(model, **kwargs)

    del model.handles
    del model.module_types

    return model

def convert_module(
            module: nn.Module,
            leave_first_and_last_layer=False,
            leave_layers: Optional[Tuple[int]] = None,
            inplace: bool = True,
            _handles: Dict = None,
            _module_types: Dict = None,
            is_root_module: bool = True,
        ) -> nn.Module:
    """ Convert module after QAT training has been done. """

    module_number = module._module_number

    if inplace == False and is_root_module:
        module = deepcopy(module)

    # get dicts from top module if not passed down recursively in this function
    _handles = module.handles if _handles is None else _handles
    _module_types = module.module_types if _module_types is None else _module_types

    if leave_first_and_last_layer and (leave_layers is None):
        leave_layers = find_first_and_last_layer(_module_types)

    assert isinstance(_handles, dict) and _handles, f"'_handles' argument needs to be dict of pre/post fwd hook handles of model modules and not {type(_handles)} {_handles}"

    #1. convert forward passes of all internal modules to handle only quantized tensors

    for name, layer in module.named_children():

        # ===== DFS down module graph ========

        try:
            convert_module(
                layer,
                leave_first_and_last_layer=leave_first_and_last_layer,
                leave_layers=leave_layers,
                _handles=_handles,
                _module_types=_module_types,
                is_root_module=False,
                inplace=True
            )
        except KeyError as KE:
            print(KE)
            print(name)
            input()

    # 2. convert known layer types and remove forward hooks on a basis of spaghetti

    # ################  remove pre / post hooks ###################

    # remove all handles

    module_handles = _handles[module_number]

    pre_handle = module_handles.get("qat_pre", None)
    if pre_handle is not None:
        pre_handle.remove() # remove fake weight quantizer : nn.Linear, Conv2d
    post_handle = module_handles.get("qat_post", None)
    if post_handle is not None:
        post_handle.remove() # remove fake activation quantizer : nn.ReLU6

    # ################  end remove pre / post hooks ###################

    dont_quantize = leave_first_and_last_layer and ( module_number in leave_layers )
    mod_type = type(module)

    is_layer = True in [issubclass(mod_type, layer) for layer in OPS]

    # delete fake quantizer:
    if hasattr(module, "_fakeQ"):
        del module._fakeQ

    if isinstance(module, QuantizableModule):

        module.quantize()

    elif is_layer and not dont_quantize:

        module.forward = _factory_convert_layer_forward_impl(module)

    return module



