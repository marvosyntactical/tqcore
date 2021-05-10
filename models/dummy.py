import torch
import torch.nn as nn
import torch.nn.functional as F

import random

check_integer_if_true = lambda x, b: (x==x.round()).all() if b else True

# convbn = ConvBNnofold

class Net(nn.Module):
    """
    Minimal example of quantizable model

    I rely on the user to: # TODO update and extend list
    1. add modules to the net exactly in the order they are called during forward (to find the next activation)
    2. that means add QuantizableResConnection after the bypassed modules
    3. add nn.Identity() after parameterized layers without non-softmax-activations (usually last layer)

    """
    def __init__(
            self,
            debug=True,
            debuginfo=(8,8,32),
            convbn=None,
            skipconv=False,
            convblocks=1,
            blocks=1,
        ): # set as argument to cnn.py

        assert convbn is not None

        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.relu1 = nn.ReLU6()
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.relu2 = nn.ReLU6()

        self.convblocks = int(convblocks)
        if skipconv:
            for i in range(self.convblocks):
                self.add_module(f"conv_{i}", nn.Sequential(
                    convbn(
                         in_planes=64,
                         out_planes=64,
                         kernel_size=3,
                         stride=1,
                         momentum=0.001,
                         relu=0,
                    ),
                    convbn(
                         in_planes=64,
                         out_planes=64,
                         kernel_size=3,
                         stride=1,
                         momentum=0.001,
                         relu=0,
                    ),
                    convbn(
                         in_planes=64,
                         out_planes=64,
                         kernel_size=3,
                         stride=1,
                         momentum=0.001,
                         relu=0,
                    ),
                    convbn(
                         in_planes=64,
                         out_planes=64,
                         kernel_size=3,
                         stride=1,
                         momentum=0.001,
                         relu=0,
                    ),
                    convbn(
                         in_planes=64,
                         out_planes=64,
                         kernel_size=3,
                         stride=1,
                         padding=4, # 5 for identity, 4 for downsample
                         momentum=0.001,
                         relu=False
                    ),
                ))

                self.add_module(f"downsample_{i}", convbn(
                     in_planes=64,
                     out_planes=64,
                     kernel_size=3,
                     padding=0,
                     stride=1,
                     momentum=0.001,
                     relu=False
                ))

                self.add_module(f"conv_res_{i}", QuantizableResConnection())
                self.add_module(f"conv_relu_{i}", nn.ReLU6())
        else:
            self.conv3 = convbn(
                in_planes=64,
                out_planes=64,
                kernel_size=3,
                stride=1,
                momentum=0.001,
                relu=0,
            )

        # self.pool = nn.MaxPool2d(2)
        self.pool = nn.AdaptiveAvgPool2d((2*7, 2*7))
        # self.pool = nn.AdaptiveAvgPool2d((1,1))

        self.fc1 = nn.Linear(12544 if skipconv else 10816, 256) # FIXME determine correct size for conv stack
        # self.fc1 = nn.Linear(64, 256) # FIXME determine correct size for conv stack
        self.relu4 = nn.ReLU6()
        self.fc2 = nn.Linear(256, 128)
        self.relu5 = nn.ReLU6()

        self.blocks = int(blocks)

        for i in range(self.blocks):

            self.add_module(f"fc_{i}", nn.Linear(128,128))
            # self.add_module(f"fc_skip_{i}", nn.Linear(128,128))

            # self.add_module(f"residual_{i}", QuantizableResConnection())
            self.add_module(f"relu_{i}", nn.ReLU6())

        self.fc3 = nn.Linear(128, 64)
        self.identity = nn.Identity()
        self.fc4 = nn.Linear(64,10)
        self.last_identity = nn.Identity()

        self.debuginfo = debuginfo
        self.skipconv = skipconv

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        if self.skipconv:
            for i in range(self.convblocks):
                exec(f"self.conv_res_{i}.cache()")
                exec(f"ds = self.downsample_{i}(x)")
                # exec(f"ds = self.conv_res_{i}.rescale(x)")
                exec(f"self.conv_res_{i}.reset()")

                exec(f"x = self.conv_res_{i}.add(ds, self.conv_{i}(x))")
                exec(f"x = self.conv_relu_{i}(x)")
        else:
            x = self.conv3(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.relu4(x)

        x = self.fc2(x)
        x = self.relu5(x)

        for i in range(self.blocks):
            # exec(f"self.residual_{i}.cache()")
            # exec(f"skippitybapbap = self.fc_skip_{i}(x)")
            # exec(f"self.residual_{i}.reset()")

            # exec(f"x = self.residual_{i}.add(skippitybapbap, self.fc_{i}(x))")
            exec(f"x = self.fc_{i}(x)")
            exec(f"x = self.relu_{i}(x)")

        x = self.fc3(x)
        x = self.identity(x) # y=x
        x = self.fc4(x)
        x = self.last_identity(x)

        # output = F.log_softmax(x, dim=1)

        return x

class NetLinear(nn.Module):
    """
    Minimal example of quantizable model with no convolutional layers, for ablation debugging conv

    I rely on the user to: # TODO update and extend list
    1. add modules to the net exactly in the order they are called during forward (to find the next activation)
    2. that means add QuantizableResConnection after the bypassed modules
    3. add nn.Identity() after parameterized layers without non-softmax-activations (usually last layer)

    """
    def __init__(self, debug=True, debuginfo=(8,8,32), conv3=False, blocks=3):
        super(NetLinear, self).__init__()
        self.blocks = blocks

        self.lin1 = nn.Linear(3072,512)
        self.relu1 = nn.ReLU6()
        self.lin2 = nn.Linear(512,64)
        self.relu2 = nn.ReLU6()
        if conv3:
            self.lin3 = nn.Linear(64,64)
            self.bn = BatchNorm1dWrap(64)

        for i in range(self.blocks):
            # right side
            self.add_module(f"fc_{i}", nn.Linear(64,64))
            self.add_module(f"bn_{i}", BatchNorm1dWrap(64))

            # left side (analog to downsample connection above)
            self.add_module(f"relu_{i}", nn.ReLU6())

        self.pool = nn.MaxPool1d(2)

        # FIXME can replace this:
        self.conv = nn.Conv2d(4,1,1,1)

        # self.fc1 = nn.Linear(64, 32)
        # self.relu4 = nn.ReLU6()

        self.fc2 = nn.Linear(8 if hasattr(self, "conv") else 32, 16)
        self.relu5 = nn.ReLU6()
        self.fc3 = nn.Linear(16, 10)
        self.identity = nn.Identity()

        self.debug = debug
        self.debuginfo = debuginfo
        self.debug_training = not self.training
        self.assertint = False

    def forward(self, x):
        p_debug = 0.01
        random_print = random.random() < p_debug
        assertint = self.assertint

        x = torch.flatten(x, 1)

        x = self.lin1(x)
        x = self.relu1(x)

        x = self.lin2(x)
        x = self.relu2(x)

        if hasattr(self, "conv3"):
            x = self.lin3(x)
            x = self.bn(x)

        for i in range(self.blocks):
            exec(f"x = self.fc_{i}(x)")
            exec(f"x = self.bn_{i}(x)")
            exec(f"x = self.relu_{i}(x)")

        # print(x.shape)
        x = self.pool(x.unsqueeze(1))
        # print(x.shape)

        # print(x.shape)

        x = x.unsqueeze(1).view(-1,4,8,1)
        x = self.conv(x)

        # print(x.shape)
        x = x.squeeze(-1).squeeze(1)

        # print(x.shape)
        # x = self.fc1(x)
        # x = self.relu4(x)

        x = self.fc2(x)
        x = self.relu5(x)
        x = self.fc3(x)
        x = self.identity(x) # y=x

        output = F.log_softmax(x, dim=1)

        return output

