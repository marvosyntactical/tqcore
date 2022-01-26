from .quantizable_layer import *

# some simple example modules

class QResTest(nn.Module):

    def __init__(self, src_dim, dim, n_labels, qkwargs, **kwargs):

        super().__init__()

        self.quant = QuantStub(**qkwargs)
        self.quantl = QListener(self.quant, plot_name="input", **qkwargs)

        self.embedding = QLinear(src_dim, dim, qkwargs=qkwargs)
        self.relu1 = QReLU6(**qkwargs)
        self.ql1 = QListener(self.embedding, self.relu1, plot_name="relu1", **qkwargs)

        self.linear = QLinear(dim, dim, qkwargs=qkwargs)
        self.relu2 = QReLU6(**qkwargs)
        self.ql2 = QListener(self.embedding, self.relu2, plot_name="relu2", **qkwargs)

        self.res = QAdd(**qkwargs)
        self.resl = QListener(self.res, plot_name="res", **qkwargs)

        self.dequant = DeQuantStub(**qkwargs)


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

        x = self.dequant(x)
        print("qrestest output:")
        print(x.shape)

        return x


