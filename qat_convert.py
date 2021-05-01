import torch

TRANSFORMER = True

if TRANSFORMER:
    from .transformer_layers import *
from .quantizable_layer import \
    ConvBNfoldable, \
    ConvBNnofold, \
    QuantizableResConnection
from .quantized_layer import \
    _factory_quantized_layer, \
    _factory_convert_relu6_layer_forward_impl,\
    _factory_convert_relu_layer_forward_impl,\
    _factory_convert_layer_forward_impl,\
    _factory_convert_quantized_identity,\
    _factory_convert_quantized_add,\
    _convert_bnnofold_layer_forward,\
    _debug_forward_pre_hook,\
    _debug_forward_post_hook,\
    _factory_convert_cache,\
    _factory_convert_reset,\
    _factory_convert_rescale,\
    _factory_non_quantized_pre_hook,\
    _factory_non_quantized_post_hook,\
    QuantizedModel

from .quantization_ema_stats import *
from .quantization_functions import *
from .qat_prepare import ACTIVS

import torch.nn as nn
import torch.nn.functional as F

import copy
from typing import Dict, Optional, Tuple
import warnings

# using lisa kuhn's code in .tinyquant/*
# to prepare a module for QAT and convert it (like in the pytorch QAT API)

# procedure is analogous to the pytorch QAT API
# (https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html#quantization-aware-training)
# 1. pretrain float model
# 2. call prep_model_qat
# 3. train model some more
# 4. call convert module

LAYERS = [
    nn.Conv2d,
    nn.Linear,
    QuantizableResConnection,
    nn.modules.batchnorm._BatchNorm,
] #, nn.MaxPool2d]

WEIGHT_LAYERS = [nn.Conv2d, nn.Linear]

def find_first_and_last_layer(module_types_dict) -> Tuple[int]:

    first_idx=0
    found_first = False
    while True:
        mod_type = module_types_dict[first_idx]
        is_layer = True in [issubclass(mod_type, layer) for layer in WEIGHT_LAYERS]
        if is_layer:
            found_first = True
            break
        first_idx += 1
    assert found_first, "did not find any implemented LAYER to pick as first layer ..."

    last_idx = max(module_types_dict.keys())
    found_last = False
    while True:
        mod_type = module_types_dict[first_idx]
        is_layer = True in [issubclass(mod_type, layer) for layer in WEIGHT_LAYERS]
        if is_layer:
            found_last = True
            break
        last_idx -= 1
    assert found_last, "did not find any implemented LAYER to pick as last layer ..."
    return (first_idx, last_idx)

def convert_module(
            module: nn.Module,
            leave_first_and_last_layer=False,
            first_and_last_layer: Optional[Tuple]=None,
            inplace: bool = True,
            _handles: Dict = None,
            _module_types: Dict = None,
            _stats = None, # TODO replace with is_root bool and just access model.stats then
        ) -> nn.Module:
    """ Convert module after QAT training has been done. """

    assert False, "reminder: implement quantization of weight matrices (pos enc, linear) HERE (not in quantized_layer"

    module_number = module._module_number

    is_root_module = _stats is None

    if is_root_module:
        module = QuantizedModel(module)

    module.register_forward_pre_hook(_debug_forward_pre_hook)

    if inplace == False and _stats is None:
        module = copy.deepcopy(module)

    # get dicts from top module if not passed down recursively in this function
    _stats = module.stats if _stats is None else _stats
    _handles = module.handles if _handles is None else _handles
    _module_types = module.module_types if _module_types is None else _module_types

    if leave_first_and_last_layer and (first_and_last_layer is None):
        first_and_last_layer = find_first_and_last_layer(_module_types)
    assert isinstance(_handles, dict) and _handles, f"'_handles' argument needs to be dict of pre/post fwd hook handles of model modules and not {type(_handles)} {_handles}"

    #1. convert forward passes of all internal modules to handle only quantized tensors

    for name, layer in module.named_children():

        # ===== DFS down module graph ========

        convert_module(
            layer,
            leave_first_and_last_layer=leave_first_and_last_layer,
            first_and_last_layer=first_and_last_layer,
            _handles=_handles,
            _module_types=_module_types,
            _stats=_stats,
            inplace=True,
        )

    # 2. convert known layer types and remove forward hooks on a basis of spaghetti

    # TODO clean up the spaghetti below

    # ################  remove pre / post hooks ###################

    # remove all handles

    module_handles = _handles[module_number]

    dont_quantize = leave_first_and_last_layer and ( module_number in first_and_last_layer) # FIXME consider adding batchnorm here
    mod_type = type(module)

    is_layer = True in [issubclass(mod_type, layer) for layer in LAYERS]
    is_activation = True in [issubclass(mod_type, activ) for activ in ACTIVS]

    if is_activation:
        module_handles["ema_handle"].remove()

    pre_handle = module_handles.get("qat_pre", None)
    if pre_handle is not None:
        pre_handle.remove() # remove fake weight quantizer : nn.Linear, Conv2d
    post_handle = module_handles.get("qat_post", None)
    if post_handle is not None:
        post_handle.remove() # remove fake activation quantizer : nn.ReLU6

    # ################  end remove pre / post hooks ###################

    # delete fake quantizer:
    if hasattr(module, "_fakeQ") and not is_activation:
        del module._fakeQ # not needed anymore

    if is_layer:

        # 1.: Find next activation

        next_activation_number = search_for_next_activ(
            _module_types,
            module_number,
        )

        print((next_activation_number-module_number, "found next activ of type", str(_module_types[next_activation_number]).split(".")[-1], "for module of type ", str(type(module)).split(".")[-1]))

        # 2.: Retrieve stats
        try:
            min_val = _stats[next_activation_number]["ema_min"]
            max_val = _stats[next_activation_number]["ema_max"]
        except KeyError:
            warnings.warn(f"Module {module_number} of type {type(module)} was never activated during QAT... make sure its not actually used!\n...", UserWarning)
            min_val, max_val = 100, 100

        if dont_quantize: # optional, for first and last layer
            module.register_forward_pre_hook(
                _factory_non_quantized_pre_hook(module)
            )
            module.register_forward_hook(
                _factory_non_quantized_post_hook(module, min_val, max_val)
            )

        else:
            # (batchnorm forward is not used in the quantized variant)
            module.forward = _factory_convert_layer_forward_impl(
               module, min_val, max_val
            )

            module.register_forward_hook(_debug_forward_post_hook)

            if isinstance(module, QuantizableResConnection):

                # also convert adding function for residual connection
                module.add = _factory_convert_quantized_add(
                    module, min_val, max_val
                )
                module.identity = _factory_convert_quantized_identity(
                    module, min_val, max_val
                )
                module.rescale = _factory_convert_rescale(
                    module, min_val, max_val
                )
                module.cache = _factory_convert_cache(
                    module
                )
                module.reset = _factory_convert_reset(
                    module
                )
                module.is_quantized = True
                module.ema_min = min_val
                module.ema_max = max_val

            # module.register_forward_pre_hook(_debug_forward_pre_hook)

    elif is_activation:
        print("Activation:",type(module))
        # FIXME should only retrieve EMA vals here. also only update qparams here TODO FIXME

        min_val = _stats[module_number]["ema_min"]
        max_val = _stats[module_number]["ema_max"]

        # case by case implementation:
        if isinstance(module, nn.ReLU6):
            module.forward = _factory_convert_relu6_layer_forward_impl(module, min_val, max_val, module._num_bits_inp)
        elif isinstance(module, nn.ReLU):
            module.forward = _factory_convert_relu_layer_forward_impl(module, min_val, max_val, module._num_bits_inp)
        elif isinstance(module, nn.Identity):
            pass
        else:
            print(f"Dont yet know how to implement quantized forward pass of {type(module)} (Or handled this elsewhere (QuantizableResConnection)")

        module.register_forward_hook(_debug_forward_post_hook)

    elif isinstance(module, nn.Identity):
        pass

    elif isinstance(module, ConvBNnofold):
        _convert_bnnofold_layer_forward(module)

    elif isinstance(module, ConvBNfoldable):
        # stop simulating Batchnom fwd passes
        # module.simulate_folding_params_handle.remove()
        # module.qat_convert_by_removing_me.remove()

        # fold weights!
        # _convert_convbnfoldable_layer_forward(module)
        module.fold()


    return module

def search_for_next_activ(module_types: Dict, mod_idx: int):
    activ_idx = mod_idx + 1
    while True:
        type_next = module_types.get(activ_idx, None)
        if type_next is None:
            raise ValueError(f"Didnt find activation after layer #{mod_idx} of type {module_types[mod_idx]}. You need to insert an nn.Identity right after this module in the model definition and call it after the last layer to fix this.")
        if type_next in ACTIVS and not (type_next == QuantizableResConnection): # list of activations imported from qat_prepare
            break
        else:
            activ_idx += 1
    return activ_idx

