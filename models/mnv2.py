import torch
import torch.nn as nn
import torch.nn.functional as F

import random

from tinyquant.quantizable_layer import *

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class InvertedResidual(nn.Module):
    def __init__(
            self, inp, oup, stride, expand_ratio, norm_layer, convbn=ConvBNfoldable, relutype=6):

        super(InvertedResidual, self).__init__()

        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []

        if expand_ratio != 1:
            layers.extend([
            # pw
            convbn(
                inp,
                hidden_dim,
                kernel_size=1,
                padding="yes",
                BNclass=norm_layer,
                relu=relutype
            )])

        layers.extend(
            # dw
            [convbn(
                hidden_dim,
                hidden_dim,
                stride=stride,
                kernel_size=3,
                groups=hidden_dim,
                BNclass=norm_layer,
                padding="yes",
                relu=relutype
            ),
            # pw-linear
            convbn(
                in_planes=hidden_dim,
                out_planes=oup,
                kernel_size=1,
                BNclass=norm_layer,
                stride=1,
                relu=False
                ),
            ])
        self.conv = nn.Sequential(*layers)

        if self.use_res_connect:
            self.skip_add = QuantizableResConnection()
        self.identity = nn.Identity()

    def forward(self, x):
        if self.use_res_connect:

            self.skip_add.cache()
            x_rescale = self.skip_add.rescale(x) # call this if one of the sides contains no module
            # conved = self.conv(x)
            self.skip_add.reset()

            x = self.skip_add.add(
                x_rescale, self.conv(x)
            )
        else:
            x = self.conv(x)
        return self.identity(x)

class MobileNetV2(nn.Module):
    def __init__(
        self,
        num_features=16,
        num_classes=10,
        width_mult=1.0,
        inverted_residual_setting=None,
        round_nearest=8,
        last_channel=1280,
        dataset_channel=1,
        convbn=ConvBNfoldable,
        norm_layer=nn.BatchNorm2d,
        relutype:int=6,
        ):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        super(MobileNetV2, self).__init__()
        block = InvertedResidual

        input_channel = num_features

        if inverted_residual_setting is None:

            # this setting from:
            # https://github.com/tinyalpha/mobileNet-v2_cifar10/blob/master/network.py
            # -> 81%

            inverted_residual_setting = [
                # t, c, n, s
                [1, num_features, 1, 1],
                [6, 24, 2, 1],
                [6, 32, 3, 1],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

            # this setting from:
            # adapted imagenet settings for cifar in https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html#quantization-aware-training
            # -> 65%

            # inverted_residual_setting = [
            #     # t, c, n, s
            #     [1, num_features, 1, 1],
            #     [6, 24, 2, 2],
            #     [6, 32, 3, 2],
            #     [6, 64, 4, 2],
            #     [6, 96, 3, 1],
            #     [6, 160, 3, 2],
            #     [6, 320, 1, 1],
            # ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "and a 4-element list, got {}".format(inverted_residual_setting))

        # build first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)

        features = [
            convbn(
                dataset_channel,
                input_channel,
                kernel_size=3,
                stride=1,
                padding="yes",
                BNclass=norm_layer,
                relu=relutype
            ),
        ]

        # build inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    block(
                        input_channel,
                        output_channel,
                        stride,
                        expand_ratio=t,
                        norm_layer=norm_layer,
                        convbn=convbn,
                        relutype=relutype
                    )
                )
                input_channel = output_channel

        # build last several layers
        features.extend(
            [convbn(
                input_channel,
                self.last_channel,
                kernel_size=1,
                BNclass=norm_layer,
                relu=relutype
            )]
        )

        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        self.pool = nn.AdaptiveAvgPool2d((1,1))

        # build classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
            nn.Identity(),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):

        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)

        output = x
        return output

