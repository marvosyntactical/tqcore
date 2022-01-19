from .qtensor import QTensor

from .quantization_functions import *

# from .qat_prepare import *
# from .qat_convert import *

# from .quantized_layer import *

from .quantizable_layer import *


global OPS, NONQUANT
OPS = [
    nn.Conv2d,
    nn.Linear,
    nn.modules.batchnorm._BatchNorm,
    nn.MaxPool2d
]

# ignore subclasses of these:
NONQUANT = [
    NonQuantizableModuleWrap
]


from .batchnorm import *
from .transformer import QTSTModel
from .lstm import QTSLSTM
from .models import *

from .histogram import *
from .config import *
from .calibration import *

qat_prepare = lambda m: m.apply(lambda mod: mod.qat_prepare() if isinstance(mod, QuantizableModule) else mod)
qat_convert = lambda m: m.apply(lambda mod: mod.quantize() if isinstance(mod, QuantizableModule) else mod)

