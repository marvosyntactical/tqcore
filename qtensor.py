import torch
from torch import nn
from torch import _C
from torch.autograd import Variable
import torch.nn.functional as F

import logging
logging.basicConfig(level="INFO")
logqt = logging.getLogger("qt")

logqt.info("QTensor logging started")


class QTensor:
    r"""
    A duck typed Tensor wrapper.

    I found it's easier to act like a Tensor and implement __torch_function__
    and whatever Tensor methods are needed
    INSTEAD OF
    trying to actually subclass Tensor
    (very messy with infinite recursions and does not save time anyways
    because you still need to reimplement everything used).
    """

    def __init__(self, data, scale:float, zero:int=0, **kwargs):
        self._t = torch.as_tensor(data, **kwargs)
        self.scale = scale
        self.zero = zero

    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """
        From
        https://pytorch.org/docs/stable/notes/extending.html#extending-torch-with-a-tensor-wrapper-type

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

        if func in HANDLED_FUNCTIONS and all(
                issubclass(t, (torch.Tensor, QTensor))
                for t in types
            ):
                logqt.info("VVVVV Invoked __torch_function__ with known func VVVVV\n")
                logqt.info(func)
                logqt.info(types)
                logqt.info("\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

                return HANDLED_FUNCTIONS[func](*args, **kwargs)
        else:

            logqt.warning("VVVVV Invoked __torch_function__ with unkown func VVVVV\n")
            logqt.warning(func)
            logqt.warning(types)
            logqt.warning("\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

            tensor_args = [a._t if isinstance(a, QTensor) else a for a in args]
            ret = func(*tensor_args, **kwargs)
            return QTensor(ret, scale=args[0].scale, zero=args[0].zero)


    # ----------------- item methods for slicing ---------------
    def __delitem__(self, *args, **kwargs):
        return QTensor(self._t.__delitem__(*args, **kwargs), self.scale, self.zero)

    def __getitem__(self, *args, **kwargs):
        return QTensor(self._t.__getitem__(*args, **kwargs), self.scale, self.zero)

    def __setitem__(self, *args, **kwargs):
        return QTensor(self._t.__setitem__(*args, **kwargs), self.scale, self.zero)

    def __getattr__(self, attr):
        # for transpose, data, etc
        tensor_attr = getattr(self._t, attr)
        if isinstance(tensor_attr, torch.Tensor):
            return QTensor(tensor_attr, scale=self.scale, zero=self.zero)
        else:
            return tensor_attr

    # ------------------- CUSTOM IMPLEMENTATIONS FROM HERE ON -------------------
    # (add each to HANDLED_FUNCTIONS if they also correspond to a function in
    # the global torch namespace

    def __add__(self, other):
        assert isinstance(other, QTensor), type(other)
        result = self._t.__add__(other._t)

        new_scale = self.scale
        new_zero = self.zero

        return QTensor(result, scale=new_scale, zero=new_zero)

    def __matmul__(self, other):
        assert isinstance(other, QTensor), type(other)

        result = self._t.__matmul__(other._t)

        new_scale = self.scale
        new_zero = self.zero

        return QTensor(result, scale=new_scale, zero=new_zero)

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
        return QTensor(self._t.to(*args, **kwargs), scale=self.scale, zero=self.zero)


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
    qt = qt.to("cuda")
    print(qt)
    print(qt.device)
    print(QParameter(qt))

