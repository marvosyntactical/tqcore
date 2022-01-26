from typing import Type, Any, Callable, Union, List, Optional, Dict
import functools

import torch
import torch.nn as nn
from torch import Tensor

try:
        from torch.hub import load_state_dict_from_url
except ImportError:
        from torch.utils.model_zoo import load_url as load_state_dict_from_url

from ..quantizable_layer import *
from ..batchnorm import *

# from .._internally_replaced_utils import load_state_dict_from_url
# from ..utils import _log_api_usage_once

# NOTE: This file was copied/adapted from
# https://raw.githubusercontent.com/pytorch/vision/main/torchvision/models/resnet.py
# , the official torchvision resnet implementation.
# It contains a quantizable version of resnet.
# Specifically, in this version of resnet, only 3x3 convolutions with stride 1,
# which are not the last convolution, are quantized
inp_kernel_size = 3 # defaults: 7 in torchvision, 3 in tvm
# eps= 1.0
# eps= 1e-5 # TODO add as config parameter; should be 1e-5 if bits activ, bits weight both large, else 1.0
activ_class = nn.LeakyReLU
# activ_class = nn.GELU
# activ_class = nn.ReLU
# log_resnet = lambda *args: None
__PRINT__ = 0
log_resnet = lambda *args: print(*args) if __PRINT__ else None

__all__ = [
    "ResNet",
    "qresnet18",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "wide_resnet50_2",
    "wide_resnet101_2",
]


model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-63fe2227.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-394f9c45.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
}

def conv2d_qmaybe(
        qwrap_class: Type[nn.Module]=None,
        qconv_class: Type[nn.Module]=None,
        qkwargs: Optional[Dict] = None,
        quantize: bool = False,
        plot_name: Optional[str] = None,
        **kwargs
    ):
    if quantize:
        mod = functools.partial(qconv_class, qkwargs=qkwargs)
    else:
        mod = nn.Conv2d
    conv = mod(**kwargs)
    if quantize:
        return qwrap_class(
            qmodule=conv,
            plot_name=plot_name,
            **qkwargs
        )
    else:
        return conv


def conv3x3_qmaybe(
        in_planes: int,
        out_planes: int,
        stride: int = 1,
        groups: int = 1,
        dilation: int = 1,
        qwrap_class: Type[nn.Module]=None,
        qconv_class: Type[nn.Module]=None,
        qkwargs: Optional[Dict] = None,
        quantize: bool = False,
        plot_name: str="",
    ) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return conv2d_qmaybe(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
        plot_name=plot_name,
        quantize=quantize,
        qwrap_class=qwrap_class,
        qconv_class=qconv_class,
        qkwargs=qkwargs,
    )


def conv1x1_qmaybe(
        in_planes: int,
        out_planes: int,
        stride: int = 1,
        qwrap_class: Type[nn.Module]=None,
        qconv_class: Type[nn.Module]=None,
        qkwargs: Optional[Dict] = None,
        quantize: bool = False,
        plot_name: str="",
    ) -> nn.Conv2d:
    """1x1 convolution"""
    return conv2d_qmaybe(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=1,
        stride=stride,
        bias=False,
        plot_name=plot_name,
        quantize=quantize,
        qwrap_class=qwrap_class,
        qconv_class=qconv_class,
        qkwargs=qkwargs,
    )

class QBasicBlock(nn.Module):
    expansion: int = 1
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        qwrap_class:Type[nn.Module]=None,
        qconv_class:Type[nn.Module]=None,
        qkwargs: Optional[Dict] = None,
        quant_all_conv: bool = False,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        quantize = stride == 1 and inplanes != 512

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = QConv2d(
            in_channels=inplanes,
            out_channels=planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=1,
            bias=False,
            dilation=1,
            qkwargs=qkwargs,
        )
        self.conv1_l = QListener(self.conv1, plot_name="conv1", **qkwargs)
        # self.conv1 = conv3x3_qmaybe(inplanes, planes, stride,
        #     qwrap_class=qwrap_class, qconv_class=qconv_class, qkwargs=qkwargs,
        #     quantize=quantize or quant_all_conv)
        # self.bn1 = NonQuantizableModuleWrap(nn.BatchNorm2d(planes), **qkwargs)
        self.bn1 = QBatchNorm2d(planes, qkwargs=qkwargs)
        self.bn1_l = QListener(self.bn1, **qkwargs)

        self.relu1 = QReLU6(**qkwargs)
        self.relu1_l = QListener(self.relu1, **qkwargs)
        # self.conv2 = conv3x3_qmaybe(planes, planes,
        #     qwrap_class=qwrap_class, qconv_class=qconv_class, qkwargs=qkwargs,
        #     quantize=inplanes!=512 or quant_all_conv)
        self.conv2 = QConv2d(
            in_channels=planes,
            out_channels=planes,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
            dilation=1,
            qkwargs=qkwargs,
        )
        self.conv2_l = QListener(self.conv2, plot_name="BB conv2", **qkwargs)

        # self.bn2 = NonQuantizableModuleWrap(nn.BatchNorm2d(planes), **qkwargs)
        self.bn2 = QBatchNorm2d(planes, qkwargs=qkwargs)
        self.bn2_l = QListener(self.bn2, **qkwargs)

        qadd_layers = []

        self.downsample = downsample
        if downsample is not None:
            qadd_layers.append(self.downsample)

        self.qadd = QAdd(**qkwargs)
        qadd_layers.append(self.qadd)
        # self.qadd = NonQuantizableModuleWrap(Add(), **qkwargs)
        self.qadd_l = QListener(*qadd_layers, **qkwargs)

        self.relu2 = QReLU6(**qkwargs)
        self.relu2_l = QListener(self.relu2, **qkwargs)

        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.conv1_l(out)

        out = self.bn1(out)
        out = self.bn1_l(out)

        out = self.relu1(out)
        out = self.relu1_l(out)

        out = self.conv2(out)
        out = self.conv2_l(out)

        out = self.bn2(out)
        out = self.bn2_l(out)

        if self.downsample is not None:
            identity = self.downsample(identity)
            identity = self.qadd_l(identity)

        out = self.qadd(out, identity)
        out = self.qadd_l(out)

        out = self.relu2(out)
        out = self.relu2_l(out)

        return out

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        qwrap_class:Type[nn.Module]=None,
        qconv_class:Type[nn.Module]=None,
        qkwargs: Optional[Dict] = None,
        quant_all_conv: bool = False,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        eps = 1e-5 # if qkwargs["num_bits"] > 2 else 1.0

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3_qmaybe(inplanes, planes, stride,
            qwrap_class=qwrap_class, qconv_class=qconv_class, qkwargs=qkwargs,
            quantize=(stride == 1 and inplanes != 512) or quant_all_conv,
            plot_name="BB conv1")
        self.bn1 = norm_layer(planes, eps=eps)
        self.dbg_ql0 = QuantModuleWrap(qmodule=None, plot_name= "BB bn1 dbg", dont_fakeQ=1, **qkwargs) # for plotting only
        self.relu1 = activ_class()
        self.dbg_ql1 = QuantModuleWrap(qmodule=None, plot_name= "BB rel1 dbg", dont_fakeQ=1, **qkwargs) # for plotting only
        self.conv2 = conv3x3_qmaybe(planes, planes,
            qwrap_class=qwrap_class, qconv_class=qconv_class, qkwargs=qkwargs,
            quantize=planes!=512 or quant_all_conv,
            plot_name="BB conv2")
        self.bn2 = norm_layer(planes, eps=eps)
        self.dbg_ql2 = QuantModuleWrap(qmodule=None, plot_name= "BB bn1 dbg", dont_fakeQ=1, **qkwargs) # for plotting only
        self.relu2 = activ_class()
        self.dbg_ql3 = QuantModuleWrap(qmodule=None, plot_name= "BB rel2 dbg", dont_fakeQ=1, **qkwargs) # for plotting only
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out_ = self.conv1(x)
        out = self.bn1(out_)
        log_resnet("------------ CRITICAL BASICBLOCK START ------------")
        log_resnet("conv1 of this block is quantizable:", isinstance(self.conv1, QuantizableModule))
        # log_resnet("bn in:", out_)
        # log_resnet("bn out:", out)
        nin = out_.unique().nelement()
        log_resnet("bn in:", nin)
        log_resnet("bn out:", out.unique().nelement())
        if nin == 1:
            log_resnet(out)
            log_resnet("bn out values^")
            log_resnet(out_)
            log_resnet("bn in values^")

        log_resnet(type(self.bn1))
        log_resnet("bn out max:", out.max())
        log_resnet("bn out min:", out.min())
        # out = self.dbg_ql0(out)
        out = self.relu1(out)

        log_resnet("relu out max:", out.max())
        log_resnet("relu out min:", out.min())

        # out = self.dbg_ql1(out)
        log_resnet("------------ CRITICAL BASICBLOCK END --------------")

        # if isinstance(self.conv2, QuantizableModule) and self.conv2.stage == QuantStage.Quantized:
        #     print("forwards are quantized")

        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.dbg_ql2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out.clone() # avoids plothackfn(custom backward) + inplace op CUDA error
        out += identity
        out = self.relu2(out)
        # out = self.dbg_ql3(out)

        return out

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        qwrap_class:Type[nn.Module]=None,
        qconv_class:Type[nn.Module]=None,
        qkwargs: Optional[Dict] = None
    ) -> None:
        raise NotImplementedError(f"Bottleneck has not been treated yet (only quantized Resnet18 sofar, which does not use bottlenecks)")
        super().__init__()
        quantize = stride==1 and inplanes != 512
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width, eps=eps)
        # self.conv2 = conv3x3_qmaybe(width, width, stride, groups, dilation)
        input(f"stride={stride}")
        self.conv1 = conv3x3_qmaybe(width, width, stride, groups, dilation,
            qwrap_class=qwrap_class, qconv_class=qconv_class, qkwargs=qkwargs,
            quantize=quantize)
        self.bn2 = norm_layer(width, eps=eps)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion, eps=eps)
        self.relu = activ_class(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class QResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        qkwargs: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        # _log_api_usage_once(self)
        self._qkwargs = qkwargs

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element list/tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group

        # self.conv1 = QConv2d(
        #     in_channels=3, # RGB input channels of dataset
        #     out_channels=self.inplanes,
        #     kernel_size=inp_kernel_size,
        #     stride=2,
        #     padding=3,
        #     bias=False,
        #     qkwargs=qkwargs,
        # )
        # self.conv1_l = QListener(self.conv1, plot_name="conv1", **qkwargs)

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=self.inplanes,
            kernel_size=inp_kernel_size,
            stride=2,
            padding=3,
            bias=False
        )
        self.quantStub = QuantStub(**qkwargs)

        self.bn1 = QBatchNorm2d(self.inplanes, qkwargs=qkwargs)
        self.bn1_l = QListener(**qkwargs)
        # self.bn1 = NonQuantizableModuleWrap(nn.BatchNorm2d(self.inplanes), **qkwargs)

        self.relu = QReLU6(**qkwargs)
        self.relu_l = QListener(self.relu, plot_name="relu", **qkwargs)

        self.maxpool = NonQuantizableModuleWrap(nn.MaxPool2d(kernel_size=3, stride=2, padding=1), **qkwargs)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        # self.avgpool_fc = NonQuantizableModuleWrap(nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)))
        #     nn.Linear(512 * block.expansion, num_classes)
        # )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.deQuantStub = DeQuantStub(**qkwargs)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[QBasicBlock]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        qkwargs = self._qkwargs

        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = [
                QConv2d(
                    in_channels=self.inplanes,
                    out_channels=planes * block.expansion,
                    stride=stride,
                    kernel_size=1,
                    bias=False,
                    qkwargs=qkwargs
                )
            ]
            downsample += [QListener(*downsample, plot_name="downsample", **qkwargs)]
            downsample += [
                QBatchNorm2d(planes*block.expansion, qkwargs=qkwargs)
                # NonQuantizableModuleWrap(nn.BatchNorm2d(planes * block.expansion), **qkwargs),
            ]
            downsample += [
                QListener(downsample[-1], **qkwargs)
            ]
            downsample = nn.Sequential(*downsample)

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation,
                qkwargs=qkwargs
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    qkwargs=self._qkwargs,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]

        x = self.conv1(x)
        # x = self.conv1_l(x)

        x = self.quantStub(x)

        x = self.bn1(x)
        x = self.bn1_l(x)

        x = self.relu(x)
        x = self.relu_l(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.deQuantStub(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)




class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        qwrap_class:Type[nn.Module]=None,
        qconv_class:Type[nn.Module]=None,
        qkwargs: Optional[Dict] = None,
        quant_all_conv: bool = False

    ) -> None:
        super().__init__()
        # _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self._qwrap_class = qwrap_class
        self._qconv_class = qconv_class
        self._quant_all_conv = quant_all_conv
        self._qkwargs = qkwargs

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element list/tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        # nn.Conv2d(3, self.inplanes, kernel_size=inp_kernel_size, stride=2, padding=3, bias=False)
        self.conv1 = conv2d_qmaybe(
                qwrap_class=qwrap_class,
                qconv_class=qconv_class,
                qkwargs=qkwargs,
                quantize=quant_all_conv,
                plot_name="conv1",
                in_channels=3,
                out_channels=self.inplanes,
                kernel_size=inp_kernel_size,
                stride=2,
                padding=3,
                bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = activ_class()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        qwrap_class  = self._qwrap_class
        qconv_class = self._qconv_class
        qkwargs = self._qkwargs
        quant_all_conv = self._quant_all_conv

        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1_qmaybe(
                    self.inplanes,
                    planes * block.expansion,
                    stride=stride,
                    qwrap_class=qwrap_class,
                    qconv_class=qconv_class,
                    qkwargs=qkwargs,
                    quantize=quant_all_conv,
                    ),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer, qwrap_class=qwrap_class, qconv_class=qconv_class, qkwargs=self._qkwargs, quant_all_conv=quant_all_conv
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    qwrap_class=qwrap_class,
                    qconv_class=qconv_class,
                    qkwargs=self._qkwargs,
                    quant_all_conv=self._quant_all_conv,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        log_resnet("~~~~~~~~~~~ RESNET FWD START ~~~~~~~~~~~~")
        log_resnet("resnet input conv v")
        x = self.conv1(x)
        log_resnet("end resnet input conv ^")
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        log_resnet("~~~~~~~~~~~ RESNET FWD END ~~~~~~~~~~~~")

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
        arch: str,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        pretrained: bool,
        progress: bool,
        resnet: Type[Union[ResNet, QResNet]] = ResNet,
        qkwargs: Dict = {},
        **kwargs: Any,
    ) -> Type[Union[ResNet, QResNet]]:

    model = resnet(block, layers, qkwargs=qkwargs, **kwargs)

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model

def qresnet18(pretrained: bool = False, progress: bool = True, qkwargs: Dict = {}, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet18", QBasicBlock, [2, 2, 2, 2], pretrained, progress,
            resnet=QResNet, qkwargs=qkwargs, **kwargs)

def resnet18(pretrained: bool = False, progress: bool = True, qkwargs: Dict = {}, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, qkwargs=qkwargs, **kwargs)


def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet34", BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet101", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)


def resnet152(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet152", Bottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs)


def resnext50_32x4d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return _resnet("resnext50_32x4d", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 8
    return _resnet("resnext101_32x8d", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet("wide_resnet50_2", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet("wide_resnet101_2", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)
