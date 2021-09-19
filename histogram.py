#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
NOTE: This file has been copied and modified from
https://github.com/NVIDIA/TensorRT/blob/master/tools/pytorch-quantization/pytorch_quantization/calib/histogram.py

For info regarding whats referred to as calibration refer to the following ppt:
https://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf
"""

from collections import Counter
import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
import warnings

import torch

class HistogramCalibrator:
    """
    Altered histogram calibrator.

    API:
    * collect() collects batch histogram into histogram
    * reset() resets histogram
    * compute_argmax() performs entropy or percentile
        calibration based on arguments

    Args:
        num_bits: An integer. Number of bits of quantization.
        num_bins: An integer. Number of histograms bins. Default 2048.
        grow_method: A string. DEPRECATED. default None.
        skip_zeros: A boolean. If True, skips zeros when collecting data. Default False.
    """
    def __init__(
            self,
            num_bits,
            num_bins=2048,
            skip_zeros=False,
            unsigned: bool = True,
        ):

        self._num_bits = num_bits
        self._num_bins = num_bins
        self._skip_zeros = skip_zeros
        self._unsigned = unsigned

        self._calib_bin_edges = None
        self._calib_hist = None

        self.plot_dir = "/home/silversurfer42/Pictures/plots/hists/"
        self.plot_tmpl = self.plot_dir + "hist.png"


    def collect(self, x):
        """Collect histogram"""
        x_np = x.cpu().detach().numpy()

        # FIXME: DEBUG: discard everything above(right of) quantile from input data:
        # q = 0.5
        # _, _tmp_bins = np.histogram(x_np, bins=self._num_bins)
        # quantile = _tmp_bins[int(q*self._num_bins)]
        # x_np = x_np[x_np < quantile]

        if self._skip_zeros:
            x_np = x_np[x_np != 0]

        if self._calib_bin_edges is None and self._calib_hist is None:
            # first time it uses num_bins to compute histogram.
            self._calib_hist, self._calib_bin_edges = np.histogram(x_np, bins=self._num_bins)

            # (initial bin width and location determined by first calib tensor...arbitrary)
        else:
            # successive call to collect()
            temp_argmax = np.max(x_np)
            temp_argmin = np.min(x_np)

            if temp_argmax > self._calib_bin_edges[-1]:
                # increase the number of bins (by extending support to the right)

                # get (uniform) step size:
                width = self._calib_bin_edges[1] - self._calib_bin_edges[0]

                # add more bin edges with same step size up to inclusion of temp_argmax:
                # (NOTE: np.arange may create an extra bin after the one containing temp_argmax)
                new_right_bin_edges = np.arange(
                    self._calib_bin_edges[-1] + width,
                    temp_argmax + width,
                    width
                )
                # concatenate old bins and new bins
                self._calib_bin_edges = np.hstack((self._calib_bin_edges, new_right_bin_edges))
            begin_old = 0
            if temp_argmin < self._calib_bin_edges[0]:
                # increase the number of bins (by extending support to the left)

                # get (uniform) step size:
                width = self._calib_bin_edges[1] - self._calib_bin_edges[0]

                new_left_bin_edges = np.arange(
                    temp_argmin - width,
                    self._calib_bin_edges[0] - width,
                    width
                )
                begin_old = len(new_left_bin_edges)
                # concatenate old bins and new bins
                self._calib_bin_edges = np.hstack((new_left_bin_edges, self._calib_bin_edges))

            # add new sample to extended support:
            hist, self._calib_bin_edges = np.histogram(x_np, bins=self._calib_bin_edges)
            # add old histogram to middle part of support
            print([arr.shape for arr in [hist[begin_old:len(self._calib_hist)], self._calib_hist]])
            hist[begin_old:len(self._calib_hist)+begin_old] += self._calib_hist
            # save updated histogram
            self._calib_hist = hist

    def plot(self, mu=None, title="Histogram"):
        y = self._calib_hist
        bins = self._calib_bin_edges
        plt.hist(y, bins=bins)
        # if mu is not None:
        #     plt.plot(mu)
        plt.gca().set(title=title, ylabel="Frequency")
        plt.savefig(self.plot_tmpl)
        plt.gcf().clear()


    def reset(self):
        """Reset the collected histogram"""
        self._calib_bin_edges = None
        self._calib_hist = None

    def compute_argmax(
            self,
            method: str,
            *,
            stride: int = 1,
            start_bin: int = 128,
            percentile: float = 99.99
        ):
        """Compute the argmax from the collected histogram

        Args:
            method: A string. One of ['entropy', 'percentile']

        Keyword Arguments:
            stride: An integer. Default 1
            start_bin: An integer. Default 128
            percentils: A float number between [0, 100]. Default 99.99.

        Returns:
            argmax: a tensor
        """
        method = method.lower()
        if method == 'entropy':
            calib_argmax = self.compute_one_bound(
                stride,
                start_bin
            )
        elif method == 'percentile':
            calib_argmax = self._compute_argmax_percentile(
                percentile
            )
        else:
            raise TypeError("Unknown calibration method {}".format(method))
        return calib_argmax

    def compute_range(self, mu: float, title: str = None):
        """
        Assumes self._calib_hist is a distribution symmetric about mu.

        Args:
            mu: ema_mu from FP32 calibration stage, tracked in QListener
        Uses:
            * self.hist_calib._calib_hist
        Returns:
            min_val, max_val: lower and upper threshold of range for quantization
        """
        hist = self._calib_hist
        bin_edges = self._calib_bin_edges

        # input(f"plotted hist in {self.plot_dir}")
        self.plot(mu=mu)

        assert hist is not None, f"must calibrate before computing range"

        right_support = bin_edges >= mu
        right_hist = hist[right_support[:-1]]
        right_bin_edges = bin_edges[right_support]

        left_support = bin_edges < mu
        left_hist = hist[left_support[:-1]]
        left_bin_edges = bin_edges[left_support]

        print("~"*20)
        print("computing range for:", title)
        print(f"leftmost bin edge={bin_edges[0]}; rightmost bin edge={bin_edges[-1]}")
        print(f"right support nbins={right_support.sum()}; N={right_hist.sum()}")
        print(f"mu={mu}")
        print(f"left support nbins={left_support.sum()}; N={left_hist.sum()}")
        print("~"*20)

        # reverse left hist because compute_one_bound computes righthand side threshold
        _, threshold_right = self.compute_one_bound(
            calib_hist=right_hist,
            calib_bin_edges=right_bin_edges,
            half=True,
            side="right"
        )
        threshold_left, _ = self.compute_one_bound(
            calib_hist=left_hist,
            calib_bin_edges=left_bin_edges,
            half=True,
            side="left"
        )
        return threshold_left, threshold_right

    def compute_one_bound(
            self,
            calib_hist=None,
            calib_bin_edges=None,
            stride=1,
            half=False,
            side="right"
        ):
        """
        Calculates "argmax" rightmost XOR leftmost bin edge such that
        taking all bins in the interval from the opposite side bin
        (i.e. farthest on other side, i.e. side=right => leftmost bin)
        bin to this "argmax" bin is the interval of bins that
        minimizes the KL-Divergence of the quantized histogram from the original histogram

        Returns the other, given by histogram, bin edge also, such that both edges are in order
        """

        if calib_hist is None:
            calib_hist = self._calib_hist
        if calib_bin_edges is None:
            calib_bin_edges = self._calib_bin_edges

        # If calibrator hasn't collected any data, return none
        if calib_bin_edges is None and calib_hist is None:
            return None

        # print(f"Computation for {side}handside:\nhist={calib_hist}\nbins={calib_bin_edges}")

        def _normalize_distr(distr):
            summ = np.sum(distr)
            if summ != 0:
                distr = distr / summ

        bin_counts = calib_hist[:]
        # counts[0] = counts[1] # FIXME dont know why this was here?

        total_data = np.sum(bin_counts)

        divergences = []

        # we are quantizing to 256 values if num_bits=8 and unsigned and this is not only half of the distribution
        # for symmetric/gaussian use case and num_bits=8, this creates only 128 bits as its used for half the dist
        nbins = 1 << (self._num_bits - 1 + int(self._unsigned) - int(half))

        # exploring a RHS threshold below the number of bins makes no sense
        # because the resulting stepsize would be smaller than what the histogram recorded
        # -> only start trying RHS treshs that are at least nbins to the right of the LHS:
        start_bin = nbins

        new_density_counts = np.zeros(nbins, dtype=np.float64)

        stop_bin = len(bin_counts)

        right = "r" in side.lower()

        # print(f"range from {start_bin} to {stop_bin}")
        for i in range(start_bin, stop_bin+1, stride):
            # in each for loop iteration, recalculate KL divergence after adding one bin to the right
            new_density_counts.fill(0) # reset

            if right:
                space = np.linspace(0, i, num=nbins+1)
            else:
                space = np.linspace(stop_bin - i, stop_bin, num=nbins+1)

            digitized_space = np.digitize(range(i), space) - 1

            digitized_space[bin_counts[:i] == 0] = -1

            # if not right:
            #     # print(f"digitized_space={digitized_space}")

            for idx, digitized in enumerate(digitized_space):
                if digitized != -1:
                    new_density_counts[digitized] += bin_counts[idx]

            counter = Counter(digitized_space)
            for key, val in counter.items():
                if key != -1:
                    new_density_counts[key] = new_density_counts[key] / val

            # if not right:
            #     # print(f"new_density_counts={new_density_counts}")

            new_density = np.zeros(i, dtype=np.float64)
            for idx, digitized in enumerate(digitized_space):
                if digitized != -1:
                    new_density[idx] = new_density_counts[digitized]

            total_counts_new = np.sum(new_density) + np.sum(bin_counts[i:])
            _normalize_distr(new_density) # optional, done by entropy

            reference_density = np.array(bin_counts[:len(digitized_space)])
            reference_density[-1] += np.sum(bin_counts[i:])

            total_counts_old = np.sum(reference_density)
            _normalize_distr(reference_density) # optional, done by entropy

            if round(total_counts_new) != total_data or round(total_counts_old) != total_data:
                ints = [int(i) for i in [total_counts_new, total_data - total_counts_new, total_counts_old, total_data - total_counts_old, total_data]]
                msg = ("="*20)+"\n{}: Count mismatch!\ntotal_counts_new={}(off by {});\ntotal_counts_old={}(off by {});\ntotal_data={}\n".format(type(self), *ints) + "="*20
                # raise RuntimeError(msg)
                # warnings.warn(msg)

            kl_div = entropy(reference_density, new_density)
            divergences.append(kl_div)

        # print(f"divergences: {divergences}")
        divergences = np.array(divergences)
        last_argmin = len(divergences) - 1 - np.argmin(divergences[::-1])
        calib_argmax = calib_bin_edges[last_argmin * stride + start_bin]

        if right:
            a = calib_bin_edges[0]
            b = calib_argmax
        else:
            a = calib_argmax
            b = calib_bin_edges[-1]
        return a, b

    @staticmethod
    def _compute_argmax_percentile(percentile):
        """Returns argmax that clips the percentile fraction of collected data"""

        if percentile < 0 or percentile > 100:
            raise ValueError("Invalid percentile. Must be in range 0 <= percentile <= 100.")

        # If calibrator hasn't collected any data, return none
        if self._calib_bin_edges is None and self._calib_hist is None:
            return None

        total = self._calib_hist.sum()
        cdf = np.cumsum(calib_hist / total)
        idx = np.searchsorted(cdf, percentile / 100)
        calib_argmax = self._calib_bin_edges[idx]
        calib_argmax = torch.tensor(calib_argmax.item()) #pylint: disable=not-callable

        return calib_argmax

    def __repr__(self):
        s = "HistogramCalibrator("
        s += super(HistogramCalibrator, self).__repr__()
        s += " calib_bin_edges={_calib_bin_edges}"
        s += " calib_hist={_calib_hist})"
        return s.format(**self.__dict__)

