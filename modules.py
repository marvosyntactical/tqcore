from .quantizable_layer import *

from typing import Dict

# some simple example modules (TODO add more)

class QResTest(nn.Module):
    def __init__(
            self,
            src_dim: int = 1,
            dim: int = 1,
            n_labels: int = 1,
            time_window:int = 1,
            qkwargs: Dict = {},
            **kwargs
        ):

        super().__init__()

        self.quant = QuantStub(**qkwargs)
        self.quantl = QListener(self.quant, plot_name="input", **qkwargs)

        self.embedding = QLinear(src_dim, dim, qkwargs=qkwargs)
        self.relu1 = QReLU6(**qkwargs)
        self.ql1 = QListener(self.embedding, self.relu1, plot_name="relu1", **qkwargs)

        self.linear = QLinear(dim, dim, qkwargs=qkwargs)
        self.relu2 = QReLU6(**qkwargs)
        self.ql2 = QListener(self.linear, self.relu2, plot_name="relu2", **qkwargs)

        self.res = QAdd(**qkwargs)
        self.resl = QListener(self.res, plot_name="res", **qkwargs)

        self.flatten = nn.Flatten(1)
        self.output = QLinear(dim * time_window, n_labels, qkwargs=qkwargs)
        self.outputl = QListener(self.output, plot_name="output", **qkwargs)

        self.dequant = DeQuantStub(**qkwargs)
        self.logsoftmax = nn.LogSoftmax(dim=-1)


    def forward(self, x, *args):

        x = self.quant(x)
        x = self.quantl(x)

        x = self.embedding(x)
        x = self.relu1(x)
        x = self.ql1(x)

        skip = x

        x = self.linear(x)
        x = self.relu2(x)
        x = self.ql2(x)

        x = self.res(x, skip)
        x = self.resl(x)

        x = self.flatten(x)
        x = self.output(x)
        x = self.outputl(x)
        x = self.dequant(x)

        x = self.logsoftmax(x)

        return x


