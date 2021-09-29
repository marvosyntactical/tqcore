import torch

import matplotlib
import matplotlib.pyplot as plt

from .qtensor import QTensor
from .quantization_functions import Quantization, \
        UniformQuantization, UniformSymmetricQuantization, FakeQuant
from .config import QuantStage

import numpy as np

__PLOT__ = 1
__DEBUG__ = 0

is_integer = lambda t: ((t.round()==t).all() if t.shape else t.round()==t) if __DEBUG__ else True

def print_qt_stats(
        name,
        qtnsr,
        stage=QuantStage.Quantized,
        step=0,
        plot=None,
        p=0.01,
        num_bits=8
    ):
    plot_dir = "/home/silversurfer42/Desktop/quant_plots/qatfull/" # TODO add to cfg/CLI or read from datetime
    plot_str = plot_dir + "{}_{}_{}.png"

    bins = None
    if isinstance(qtnsr, QTensor):
        data, scale, zero = qtnsr._t, qtnsr.scale, qtnsr.zero
        if stage == QuantStage.QAT:
            # print(data)
            # print(f"Tensor unique values ^")
            # assert not qtnsr.quantized
            a = ( 0 - zero) * scale
            b = (255 - zero) * scale
            size = (b-a)/255
        else:
            vmin, vmax = data.min(), data.max()

            through = len(torch.unique(data))
            mini = min(2**num_bits, data.nelement())
            through_ratio = (through/mini) * 100

            fstr = f"STATS({name}):\nSCALE\t= {scale},\nZERO\t= {zero};"
            fstr += f"\nMIN\t= {vmin};\nMAX\t= {vmax};"
            fstr += f"\nSHAPE\t= {data.shape}"
            fstr += f"\nNELEM\t= {data.nelement()}"
            fstr += f"\nUNIQ\t= {data.unique()}"
            fstr += f"\n#UNIQ\t= {through} ({through_ratio}% of {mini})"
            print("="*20)
            print(fstr)

            # assert qtnsr.quantized
            a = 0
            b = 255
            size = 1
        # get bins
        bins = np.arange(a,b+size,size)
        info = f",scale={round(scale,3)}, zero={round(zero,3)}"
    else:
        bins = 1000
        data = qtnsr
        info = ""

        # Original idea to investigate binning:
        # sorted_data = data.reshape(-1).sort()[0]
        # shifts = (sorted_data - sorted_data.roll(1))
        # weighted_var = shifts.var()/(data.max()-data.min())
        # print(f"WEIGHTED_VAR({name}): {weighted_var.item()}")

    # sometimes show matplotlib histogram
    if __PLOT__ and torch.rand(1) <= p: # and stage==QuantStage.Quantized:
        plot_data = data.detach().reshape(-1).numpy()
        plt.hist(plot_data, bins=bins)
        stage_ = "QAT" if stage == QuantStage.QAT else ("FP32" if stage==QuantStage.FP32 else \
                ("Calib" if stage == QuantStage.Calibration else "Quantized"))
        plt.gca().set(title=stage_+f" histogram of {name} at batch #{step}"+info, ylabel="Freq")
        plt.savefig(plot_str.format(stage_, name, step))
        plt.gcf().clear()




