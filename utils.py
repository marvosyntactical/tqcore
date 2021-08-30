import torch

import matplotlib
import matplotlib.pyplot as plt

from .qtensor import QTensor
from .quantization_functions import Quantization, \
        UniformQuantization, UniformSymmetricQuantization, FakeQuant
from .config import QuantStage


__PLOT__ = 0
__DEBUG__ = 0

is_integer = lambda t: ((t.round()==t).all() if t.shape else t.round()==t) if __DEBUG__ else True

def print_qt_stats(
        name,
        qtnsr,
        stage=QuantStage.Quantized,
        step=0,
        plot=None
    ):
    plot_dir = "/home/silversurfer42/Pictures/plots/again/"
    qat_plot_str = plot_dir + "QAT_{}_{}.png"
    fp_plot_str =  plot_dir + "FP32_{}_{}.png"
    if stage == QuantStage.Quantized:
        data, scale, zero = qtnsr._t, qtnsr.scale, qtnsr.zero
        vmin, vmax = data.min(), data.max()
        assert qtnsr.quantized

        num_bits = 8
        through = len(torch.unique(data))
        through_ratio = (through/min(2**num_bits, data.nelement())) * 100

        fstr = f"STATS({name}):\nSCALE\t= {scale},\nZERO\t= {zero};"
        fstr += f"\nMIN\t= {vmin};\nMAX\t= {vmax};"
        fstr += f"\nSHAPE\t= {data.shape}"
        fstr += f"\nNELEM\t= {data.nelement()}"
        fstr += f"\nUNIQ\t= {data.unique()}"
        fstr += f"\n#UNIQ\t= {through} ({through_ratio}%)"
        print("="*20)
        print(fstr)

    elif (stage == QuantStage.QAT) and __DEBUG__:
        if isinstance(qtnsr, QTensor):
            assert not qtnsr.quantized
            data = qtnsr._t
        else:
            data = qtnsr

        sorted_data = data.reshape(-1).sort()[0]
        shifts = (sorted_data - sorted_data.roll(1))
        weighted_var = shifts.var()/(data.max()-data.min())
        print(f"WEIGHTED_VAR({name}): {weighted_var.item()}")

        # sometimes show matplotlib histogram
        if __PLOT__ and torch.rand(1) > 0.9:
            plot_data = data.detach().reshape(-1).numpy()
            plt.hist(plot_data, bins=100)
            plt.gca().set(title=f"QAT Activation hist after {name}", ylabel="Freq")
            plt.savefig(QAT_plot_str.format(name, step))
            plt.gcf().clear()

    elif (stage == QuantStage.FP32):
        data = qtnsr
        # sometimes show matplotlib histogram
        if __PLOT__ and torch.rand(1) > 0.9 or plot=="always":
            plot_data = data.detach().reshape(-1).numpy()
            plt.hist(plot_data, bins=100)
            plt.gca().set(title=f"FP32 Activation hist after {name}", ylabel="Freq")
            plt.savefig(fp_plot_str.format(name, step))
            plt.gcf().clear()


class QuantConfigurationError(Exception):
    pass


