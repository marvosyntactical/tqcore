import torch
from torch import nn
from torch import Tensor
from torch import _C
from torch.autograd import Variable
import torch.nn.functional as F

from copy import deepcopy
from typing import Callable, List, Dict, Tuple, Optional

import logging
logqt = logging.getLogger("QTensor")

__all__ = ["QTensor"]

__DEBUG__ = 0
__LOG__ = 0

if __LOG__:
    logqt.info("QTensor logging started")
else:
    logging.disable()


class QTensor:
    r"""
    A duck typed Tensor wrapper.

    Torch allows both subclassing Tensor and doing duck typing like here;
    I found it's easier to act like a Tensor and implement __torch_function__
    and whatever Tensor methods are needed
    INSTEAD OF
    trying to actually subclass Tensor
    (very messy with infinite recursions and does not save coding time anyways
    because you still need to reimplement everything used).


    QTensor has two use cases:
        * not quantized: used during quantization aware training to hold scale and zero point and so on for debugging purposes; holds FP32 or rather fake quantized data
        * quantized: after model conversion; actually holds quantized data and methods such as QTensor.dequantize() may be used
    """

    def __init__(
            self,
            data,
            scale: float,
            zero: int=0,
            quantized: bool=True,
            symmetric:bool=False,
            num_bits: Optional[int]=None,
            **kwargs
        ):
        # TODO add num bits attribute for architectures with activations with differing bitwidths
        assert num_bits is not None
        self.num_bits = num_bits

        self._t: Tensor = torch.as_tensor(data, **kwargs)
        self.scale: float = scale
        self.zero: int = zero
        self.quantized: bool = bool(quantized)
        self.symmetric: bool = bool(symmetric)
        # overwrite attributes here to return Tensor, otherwise __getattr__ attempts to return QTensor
        self.shape: Tensor = self._t.shape

        if self.quantized and not self._t.dtype==torch.bool:
            assert torch.allclose(self._t, self._t.round()), f"quantized QTensor should only be initialized with already quantized, that is rounded, data, but got: {data}"
            if not symmetric and __DEBUG__:
                assert (self._t >= self.zero).all(), (self._t.min(), self.zero)

    def dequantize(self) -> Tensor:
        assert self.quantized, "may only dequantize QTensor holding quantized values; use qtensor._t instead"
        return (self._t - self.zero) * self.scale

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """
        From
        https://pytorch.org/docs/stable/notes/extending.html#extending-torch-with-a-tensor-wrapper-type

        Original DocString:
        This __torch_function__ implementation wraps subclasses such that
        methods called on subclasses return a subclass instance instead of
        a ``torch.Tensor`` instance.
        One corollary to this is that you need coverage for torch.Tensor
        methods if implementing __torch_function__ for subclasses.
        We recommend always calling ``super().__torch_function__`` as the base
        case when doing the above.
        While not mandatory, we recommend making `__torch_function__` a classmethod.
        """
        if kwargs is None:
            kwargs = {}

        quantized_or_not = [a.quantized for a in args if isinstance(a, QTensor)]
        # assert len(set(quantized_or_not)) == 1, (func, [type(a) for a in args]) # either all quantized or none quantized

        if func in HANDLED_FUNCTIONS and all(
                issubclass(t, (torch.Tensor, QTensor))
                for t in types
            ):
                logqt.debug("VVVVV Invoked QTensor __torch_function__ with known func VVVVV\n")
                logqt.debug(func)
                logqt.debug(types)
                logqt.debug("\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

                return HANDLED_FUNCTIONS[func](*args, **kwargs)

        elif func in PROHIBITED_FUNCTIONS:

            logqt.warning("VVVVV Invoked QTensor __torch_function__ with prohibited func VVVVV\n")
            logqt.warning(func)
            logqt.warning(types)
            logqt.warning("There is another way to accomplish this operation:")
            logqt.warning(PROHIBITED_FUNCTIONS[func])
            logqt.error("\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

        else:

            logqt.warning("VVVVV Invoked QTensor __torch_function__ with unkown func VVVVV\n")
            logqt.warning(func)
            logqt.warning(types)
            logqt.warning("\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

            qtargs = [a for a in args if isinstance(a, QTensor)]

            assert all(a.quantized for a in qtargs) or all(not a.quantized for a in qtargs), \
                    [a.quantized for a in qtargs]
            assert len(set(a.num_bits for a in qtargs)) == 1, \
                    [a.num_bits for a in qtargs]

            tensor_args = [a._t if isinstance(a, QTensor) else a for a in args]
            ret = func(*tensor_args, **kwargs)
            return QTensor(
                ret,
                scale=args[0].scale,
                zero=args[0].zero,
                quantized=args[0].quantized,
                num_bits=args[0].num_bits
            )

    # ----------------- item methods for tensor slicing ---------------
    def __delitem__(self, *args, **kwargs):
        return QTensor(self._t.__delitem__(*args, **kwargs), self.scale, self.zero,
                quantized=self.quantized, num_bits=self.num_bits)

    def __getitem__(self, *args, **kwargs):
        return QTensor(self._t.__getitem__(*args, **kwargs), self.scale, self.zero,
                quantized=self.quantized, num_bits=self.num_bits)

    def __setitem__(self, *args, **kwargs):
        return QTensor(self._t.__setitem__(*args, **kwargs), self.scale, self.zero,
                quantized=self.quantized, num_bits=self.num_bits)

    def __getattr__(self, attr):
        # for transpose(), etc:
        # default behavior: return QTensor.
        # must implement attrs/methods that shouldntt return QTensors, e.g. .size()
        # when they are implemented, __getattr__ is never called for them
        tensor_attr = getattr(self._t, attr)

        if isinstance(tensor_attr, torch.Tensor):
            return QTensor(tensor_attr, scale=self.scale, zero=self.zero, quantized=self.quantized, num_bits=self.num_bits)

        elif isinstance(tensor_attr, Callable):
            # NOTE: assuming all Tensor methods return Tensors
            # (this assumption may be wrong,
            # => look out for counterexamples and implement them as a method of QTensor separately)

            # list of methods that dont return Tensor:
            # Tensor.size(dim) -> int (use Tensor.shape instead)

            logqt.warning("VVVVV Invoked QTensor.__getattr__ with unkown Tensor method VVVVV\n")
            logqt.warning(attr)
            logqt.warning("\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

            method_returning_qtensor = lambda *args, **kwargs: \
                QTensor(tensor_attr(*args, **kwargs), scale=self.scale, zero=self.zero,
                        quantized=self.quantized, num_bits=self.num_bits)
            return method_returning_qtensor
        else:
            return tensor_attr

    # ------------------- CUSTOM IMPLEMENTATIONS FROM HERE ON -------------------
    # (add each to HANDLED_FUNCTIONS if they also correspond to a function in
    # the global torch namespace

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            result = type(self)(self._t.clone(memory_format=torch.preserve_format), scale=self.scale, zero=self.zero, quantized=self.quantized, num_bits=self.num_bits)
            memo[id(self)] = result
            return result

    def __repr__(self):
        string = str(self._t)
        string = "QT"+string[1:-1]
        string += ", scale="+str(self.scale)
        string += ", zero="+str(self.zero)
        string += f", quantized={self.quantized}"
        string += f", num_bits={self.num_bits}"
        string += ")"
        return string

    def split(self, *args, **kwargs):
        tup_out: Tuple[Tensor] = self._t.split(*args, **kwargs)
        outs: List = [QTensor(
            o,
            scale=self.scale, zero=self.zero,
            quantized=self.quantized,
            symmetric=self.symmetric,
            num_bits=self.num_bits
        ) for o in tup_out]
        return tuple(outs)

    def to(self, *args, **kwargs):
        return QTensor(self._t.to(*args, **kwargs), scale=self.scale, zero=self.zero,
                quantized=self.quantized, symmetric=self.symmetric, num_bits=self.num_bits)

    # ------------------- The following methods do not return QTensors -------------------
    def size(self, *args, **kwargs) -> Tensor:
        return self._t.size(*args, **kwargs)

    def dim(self, *args, **kwargs) -> int:
        # tensor rank
        return self._t.dim(*args, **kwargs)

    def item(self, *args, **kwargs) -> object:
        return self._t.item(*args, **kwargs)

    def nelement(self, *args, **kwargs) -> int:
        return self._t.nelement(*args, **kwargs)

    def any(self, *args, **kwargs) -> bool:
        return self._t.any(*args, **kwargs)

HANDLED_FUNCTIONS = {
}

PROHIBITED_FUNCTIONS = {
    torch.split: "Use QTensor.split instead",
    torch.stack: "Use tqcore.quantizable_layer.QStack instead",
    torch.cat: "Use tqcore.quantizable_layer.QCat instead",
    torch.add: "Use tqcore.kernel.qadd instead",
    torch.matmul: "Use tqcore.kernel.qmul instead",
    torch.mul: "Use tqcore.kernel.qmul instead",
}


