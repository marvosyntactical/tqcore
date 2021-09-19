import torch

import matplotlib
import matplotlib.pyplot as plt

from .qtensor import QTensor
from .quantization_functions import Quantization, \
        UniformQuantization, UniformSymmetricQuantization, FakeQuant
from .config import QuantStage


__PLOT__ = 1
__DEBUG__ = 0

is_integer = lambda t: ((t.round()==t).all() if t.shape else t.round()==t) if __DEBUG__ else True

class QuantConfigurationError(Exception):
    pass

def print_qt_stats(
        name,
        qtnsr,
        stage=QuantStage.Quantized,
        step=0,
        plot=None,
        p=0.2
    ):
    plot_dir = "/home/silversurfer42/Desktop/quant_plots/sep1/" # TODO add to cfg/CLI or read from datetime
    calib_plot_str = plot_dir + "Calib_{}_{}.png"
    qat_plot_str = plot_dir + "QAT_{}_{}.png"
    fp_plot_str =  plot_dir + "FP32_{}_{}.png"
    plot_str = qat_plot_str if stage==QuantStage.QAT else (calib_plot_str if stage==QuantStage.Calibration else fp_plot_str)
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

    else:
        if isinstance(qtnsr, QTensor):
            assert not qtnsr.quantized
            data = qtnsr._t
        else:
            data = qtnsr

        # Original idea to investigate binning:
        # sorted_data = data.reshape(-1).sort()[0]
        # shifts = (sorted_data - sorted_data.roll(1))
        # weighted_var = shifts.var()/(data.max()-data.min())
        # print(f"WEIGHTED_VAR({name}): {weighted_var.item()}")

        # sometimes show matplotlib histogram
        if __PLOT__ and torch.rand(1) < p:
            plot_data = data.detach().reshape(-1).numpy()
            plt.hist(plot_data, bins=100)
            stage_ = "QAT" if stage == QuantStage.QAT else ("FP32" if stage==QuantStage.FP32 else "Calib")
            plt.gca().set(title=stage_+f" Activation hist after {name}", ylabel="Freq")
            plt.savefig(plot_str.format(name, step))
            plt.gcf().clear()




