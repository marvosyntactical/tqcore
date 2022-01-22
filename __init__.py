from .qtensor import QTensor

from .quantization_functions import *
from .quantizable_layer import *
from .batchnorm import *
from .transformer import QTSTModel
from .lstm import QTSLSTM
from .modules import *
from .models import *

from .histogram import *
from .config import *
from .calibration import *

qat_prepare = lambda m: m.apply(lambda mod: mod.qat_prepare() if isinstance(mod, QuantizableModule) else mod)
qat_convert = lambda m: m.apply(lambda mod: mod.quantize() if isinstance(mod, QuantizableModule) else mod)

