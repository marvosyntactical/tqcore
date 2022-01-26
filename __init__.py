from .qtensor import QTensor

from .quantization_functions import *
<<<<<<< HEAD

# from .qat_prepare import *
# from .qat_convert import *

# from .quantized_layer import *
=======
>>>>>>> 5d0d38620c339bb3dbf4166ed543fbadf66e7b5f
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

