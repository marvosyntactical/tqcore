import torch
from torch import nn

import warnings

from .quantization_functions import *
from .quantizable_layer import QuantizableModule
from .quantized_layer import _factory_convert_layer_forward_impl, \
        OPS, NONQUANT
from .qat_convert import *

def calibration_prepare(
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

    warnings.warn(f"TODO Implement Calibration preparation")

    if inplace == False and is_root_module:
        module = copy.deepcopy(module)

    mod_type = type(module)

    descendants_module_number = module_number + 1

    is_nonquantizable = issubclass(mod_type, tuple(NONQUANT))
    named_children = dict(module.named_children())

    if is_nonquantizable:
        # do not recurse to children
        del named_children["fp_module"]

    for name, layer in named_children.items():
        _, descendants_module_number = calibration_prepare(
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
    # -------- PREPARE DIFFERENT MODULES CASE BY CASE -------- #
    param_names = [name for name, _ in module.named_parameters()]

    if isinstance(module, QuantizableModule):
        module.calibration_prepare()
    elif isinstance(module, nn.Linear) or \
           isinstance(module, nn.Conv2d) or \
           isinstance(module, nn.modules.batchnorm._BatchNorm):

        assert not set(param_names) - {"weight", "bias"}, param_names

        module._Qwt = quant_weight()
        module._num_bits_wt = num_bits_weight
        module._num_bits_bias = num_bits_bias

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

    return module, descendants_module_number

def calibration_convert(
            module: nn.Module,
            leave_first_and_last_layer=False,
            leave_layers: Optional[Tuple[int]] = None,
            _module_types: Dict = None,
            inplace: bool = True,
            is_root_module: bool = True,
        ) -> nn.Module:
    """ Convert module after QAT training has been done. """

    if inplace == False and is_root_module:
        module = deepcopy(module)

    if leave_first_and_last_layer and (leave_layers is None):
        raise NotImplementedError(f"leave first and last layer not yet implemented for calibration")
        leave_layers = find_first_and_last_layer(_module_types)

    mod_type = type(module)

    is_nonquantizable = True in [issubclass(mod_type, layer) for layer in NONQUANT]

    named_children = dict(module.named_children())

    if is_nonquantizable:
        # do not recurse to children
        del named_children["fp_module"]

    #1. convert forward passes of all internal modules to handle only quantized tensors

    for name, layer in named_children.items():

        # ===== DFS down module graph ========

        calibration_convert(
            layer,
            leave_first_and_last_layer=leave_first_and_last_layer,
            leave_layers=leave_layers,
            _module_types=_module_types,
            is_root_module=False,
            inplace=True
        )

    # 2. convert known layer types

    dont_quantize = leave_first_and_last_layer and ( module_number in leave_layers )

    is_layer = True in [issubclass(mod_type, layer) for layer in OPS]

    if isinstance(module, QuantizableModule):
        module.quantize()
    elif is_layer and not dont_quantize:
        module.forward = _factory_convert_layer_forward_impl(module)
    return module



