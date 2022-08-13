# TinyQuant Core

A Small Framework for Simulation of Low-Bitwidth Quantization in Pytorch.

Pytorch itself, as of August 2022, contains various quantization options, however None for Below ```8bit```.

This is a small framework for testing out the loss in accuracy when quantizing a pytorch model to below 8bit.

It is NOT aimed at efficiency (Therefore it is implemented in raw pytorch, not CUDA/C++).

The Quantization Framework used is that of [Jacob et al. (2018)](https://openaccess.thecvf.com/content_cvpr_2018/html/Jacob_Quantization_and_Training_CVPR_2018_paper.html).

It was presented for CNNs in that paper, and is applied to the transformer here; with various options.

``` 
@inproceedings{jacob2018quantization,
  title={Quantization and training of neural networks for efficient integer-arithmetic-only inference},
  author={Jacob, Benoit and Kligys, Skirmantas and Chen, Bo and Zhu, Menglong and Tang, Matthew and Howard, Andrew and Adam, Hartwig and Kalenichenko, Dmitry},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={2704--2713},
  year={2018}
}
``` 


Adapted from [bklein/dl-frameworks-quantization](https://cegit.ziti.uni-heidelberg.de/bklein/dl-frameworks-quantization)



