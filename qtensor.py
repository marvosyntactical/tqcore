import torch
from torch import nn
from torch import Tensor
from torch import _C
from torch.autograd import Variable
import torch.nn.functional as F

from copy import deepcopy
from typing import Callable

import logging
logging.basicConfig(level="INFO")
logqt = logging.getLogger("qt")

logging.disable()
# logqt.info("QTensor logging started")

class QTensor:
    r"""
    A duck typed Tensor wrapper.

    I found it's easier to act like a Tensor and implement __torch_function__
    and whatever Tensor methods are needed
    INSTEAD OF
    trying to actually subclass Tensor
    (very messy with infinite recursions and does not save coding time anyways
    because you still need to reimplement everything used).
    """

    def __init__(self, data, scale: float, zero: int=0, quantized: bool=True, symmetric:bool=False, **kwargs):
        self._t = torch.as_tensor(data, **kwargs)
        self.scale = scale
        self.zero = zero
        self.quantized = bool(quantized)
        self.symmetric = symmetric

        if self.quantized:
            # TODO remove this costly assertion after testing !!!!! FIXME:
            assert torch.allclose(self._t, self._t.round()), f"QTensor should only be initialized with already quantized, that is rounded, data, but got: {data}"
            if not symmetric:
                assert (self._t >= self.zero).all(), (self._t.min(), self.zero)

        # overwrite attributes here to return Tensor, otherwise __getattr__ attempts to return QTensor
        self.shape: Tensor = self._t.shape

    def dequantize(self) -> Tensor:
        return (self._t - self.zero) * self.scale

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
        assert len(set(quantized_or_not)) == 1, [type(a) for a in args] # either all quantized or none quantized

        if func in HANDLED_FUNCTIONS and all(
                issubclass(t, (torch.Tensor, QTensor))
                for t in types
            ):
                logqt.info("VVVVV Invoked QTensor __torch_function__ with known func VVVVV\n")
                logqt.info(func)
                logqt.info(types)
                logqt.info("\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

                return HANDLED_FUNCTIONS[func](*args, **kwargs)
        else:

            logqt.warning("VVVVV Invoked QTensor __torch_function__ with unkown func VVVVV\n")
            logqt.warning(func)
            logqt.warning(types)
            logqt.warning("\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

            tensor_args = [a._t if isinstance(a, QTensor) else a for a in args]
            ret = func(*tensor_args, **kwargs)
            return QTensor(ret, scale=args[0].scale, zero=args[0].zero,
                    quantized=args[0].quantized)


    # ----------------- item methods for slicing ---------------
    def __delitem__(self, *args, **kwargs):
        return QTensor(self._t.__delitem__(*args, **kwargs), self.scale, self.zero,
                quantized=self.quantized)

    def __getitem__(self, *args, **kwargs):
        return QTensor(self._t.__getitem__(*args, **kwargs), self.scale, self.zero,
                quantized=self.quantized)

    def __setitem__(self, *args, **kwargs):
        return QTensor(self._t.__setitem__(*args, **kwargs), self.scale, self.zero,
                quantized=self.quantized)

    def __getattr__(self, attr):
        # for transpose(), data, etc:
        # default behavior: return QTensor.
        # must implement attrs/methods that shant return QTensors, e.g. .size()
        # when they are implemented, __getattr__ is never called for them
        tensor_attr = getattr(self._t, attr)

        if isinstance(tensor_attr, torch.Tensor):
            return QTensor(tensor_attr, scale=self.scale, zero=self.zero)

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
                        quantized=self.quantized)
            return method_returning_qtensor
        else:
            return tensor_attr

    # ------------------- CUSTOM IMPLEMENTATIONS FROM HERE ON -------------------
    # (add each to HANDLED_FUNCTIONS if they also correspond to a function in
    # the global torch namespace

    def __add__(self, other):
        assert isinstance(other, QTensor), type(other)

        # NEED TO SCALE TENSORS DOWN BEFORE CALLING THIS

        # actually calculate in FP (could switch to custom backend with int8 kernel here in future)
        r = self._t + other._t

        new_scale = self.scale + other.scale
        new_zero = self.zero + other.zero

        return QTensor(r, scale=new_scale, zero=new_zero,
                quantized=self.quantized)

    def __matmul__(self, other):
        assert isinstance(other, QTensor), type(other)

        r = self._t @ other._t

        new_scale = self.scale + other.scale
        new_zero = self.zero + other.zero

        return QTensor(r, scale=new_scale, zero=new_zero,
                quantized=self.quantized)

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            result = type(self)(self._t.clone(memory_format=torch.preserve_format), scale=self.scale, zero=self.zero)
            memo[id(self)] = result
            return result

    def __repr__(self):
        string = str(self._t)
        string = "QT"+string[1:-1]
        string += ", scale="+str(self.scale)
        string += ", zero="+str(self.zero)
        string += ")"
        return  string

    def to(self, *args, **kwargs):
        return QTensor(self._t.to(*args, **kwargs), scale=self.scale, zero=self.zero,
                quantized=self.quantized)

    def size(self, *args, **kwargs):
        return self._t.size(*args, **kwargs)

    def dim(self, *args, **kwargs):
        return self._t.dim(*args, **kwargs)



HANDLED_FUNCTIONS = {
    torch.add: QTensor.__add__,
    torch.matmul: QTensor.__matmul__
}


#  DELETE THE BELOW, ONLY FOR TESTING:
if __name__ == "__main__":
    qt = QTensor(torch.rand(3,5), scale=0.0015,zero=1)
    print(qt)
    print(qt[1:])
    print(qt+qt)
    print(torch.add(qt,qt))
    trans = torch.transpose(qt,0,1)
    print(trans)
    print(torch.matmul(qt, trans))
    print(qt.T)
    qt = qt.to("cpu")
    print(qt)
    print(qt.device)
    print(deepcopy(qt))

    from functools import wraps
    import time

    def timeit(my_func):
        @wraps(my_func)
        def timed(*args, **kw):

            tstart = time.time()
            output = my_func(*args, **kw)
            tend = time.time()

            print('"{}" took {:.3f} s to execute\n'.format(my_func.__name__, (tend - tstart)))
            return output
        return timed

    @timeit
    def torch_softmax(inp):
        return inp.softmax(dim=-1)

    @timeit
    def tqcore_softmax(inp):
        return



