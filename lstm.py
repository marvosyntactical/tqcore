"""
Implementation of quantized batch-normalized LSTM.
This file is from
jinhunchoi's implementation of BNLSTM at
https://raw.githubusercontent.com/jihunchoi/recurrent-batch-normalization-pytorch/master/bnlstm.py
and extended with some more LSTM wrapper modules



# NOTE TODO
Implement QBNLSTM analogously to the below:
https://github.com/quic/aimet-model-zoo/blob/develop/zoo_torch/examples/deepspeech2_quanteval.py
"""


import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional, init
from utils import xavier_uniform


class SeparatedBatchNorm1d(nn.Module):

    """
    A batch normalization module which keeps its running mean
    and variance separately per timestep.
    """

    def __init__(self, num_features, max_length, eps=1e-5, momentum=0.1,
                 affine=True):
        """
        Most parts are copied from
        torch.nn.modules.batchnorm._BatchNorm.
        """

        super(SeparatedBatchNorm1d, self).__init__()
        self.num_features = num_features
        self.max_length = max_length
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        if self.affine:
            self.weight = nn.Parameter(torch.FloatTensor(num_features))
            self.bias = nn.Parameter(torch.FloatTensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        for i in range(max_length):
            self.register_buffer(
                'running_mean_{}'.format(i), torch.zeros(num_features))
            self.register_buffer(
                'running_var_{}'.format(i), torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.max_length):
            running_mean_i = getattr(self, 'running_mean_{}'.format(i))
            running_var_i = getattr(self, 'running_var_{}'.format(i))
            running_mean_i.zero_()
            running_var_i.fill_(1)
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input_):
        if input_.size(1) != self.running_mean_0.nelement():
            raise ValueError('got {}-feature tensor, expected {}'
                             .format(input_.size(1), self.num_features))

    def forward(self, input_, time):
        self._check_input_dim(input_)
        if time >= self.max_length:
            time = self.max_length - 1
        running_mean = getattr(self, 'running_mean_{}'.format(time))
        running_var = getattr(self, 'running_var_{}'.format(time))
        return functional.batch_norm(
            input=input_, running_mean=running_mean, running_var=running_var,
            weight=self.weight, bias=self.bias, training=self.training,
            momentum=self.momentum, eps=self.eps)

    def __repr__(self):
        return ('{name}({num_features}, eps={eps}, momentum={momentum},'
                ' max_length={max_length}, affine={affine})'
                .format(name=self.__class__.__name__, **self.__dict__))


class LSTMCell(nn.Module):

    """A basic LSTM cell."""

    def __init__(self, input_size, hidden_size, use_bias=True, **kwargs):
        """
        Most parts are copied from torch.nn.LSTMCell.
        """

        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.weight_ih = nn.Parameter(
            torch.FloatTensor(input_size, 4 * hidden_size))
        self.weight_hh = nn.Parameter(
            torch.FloatTensor(hidden_size, 4 * hidden_size))
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(4 * hidden_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters following the way proposed in the paper.
        """

        init.orthogonal(self.weight_ih.data)

        weight_hh_data = torch.eye(self.hidden_size)
        weight_hh_data = weight_hh_data.repeat(1, 4)

        with torch.no_grad():
            self.weight_hh.set_(weight_hh_data)

        # The bias is just set to zero vectors.
        if self.use_bias:
            init.constant(self.bias.data, val=0)

    def forward(self, input_, hx):
        """
        Args:
            input_: A (batch, input_size) tensor containing input
                features.
            hx: A tuple (h_0, c_0), which contains the initial hidden
                and cell state, where the size of both states is
                (batch, hidden_size).

        Returns:
            h_1, c_1: Tensors containing the next hidden and cell state.
        """

        h_0, c_0 = hx
        batch_size = h_0.size(0)
        bias_batch = self.bias.unsqueeze(0).expand(
            batch_size, *self.bias.size()
        )

        wh_b = torch.addmm(bias_batch, h_0, self.weight_hh)
        wi = torch.mm(input_, self.weight_ih)
        f, i, o, g = torch.split(wh_b + wi,
                                 self.hidden_size, dim=1)
        c_1 = torch.sigmoid(f) * c_0 + torch.sigmoid(i) * torch.tanh(g)
        h_1 = torch.sigmoid(o) * torch.tanh(c_1)
        return h_1, c_1

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class BNLSTMCell(nn.Module):

    """A BN-LSTM cell."""

    def __init__(self, input_size, hidden_size, max_length, use_bias=True):

        super(BNLSTMCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.use_bias = use_bias
        self.weight_ih = nn.Parameter(
            torch.FloatTensor(input_size, 4 * hidden_size))
        self.weight_hh = nn.Parameter(
            torch.FloatTensor(hidden_size, 4 * hidden_size))

        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(4 * hidden_size))
        else:
            self.register_parameter('bias', None)

        # BN parameters
        self.bn_ih = SeparatedBatchNorm1d(
            num_features=4 * hidden_size, max_length=max_length)
        self.bn_hh = SeparatedBatchNorm1d(
            num_features=4 * hidden_size, max_length=max_length)
        self.bn_c = SeparatedBatchNorm1d(
            num_features=hidden_size, max_length=max_length)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters following the way proposed in the paper.
        """

        # The input-to-hidden weight matrix is initialized orthogonally.
        init.orthogonal(self.weight_ih.data)
        # The hidden-to-hidden weight matrix is initialized as an identity
        # matrix.
        weight_hh_data = torch.eye(self.hidden_size)
        weight_hh_data = weight_hh_data.repeat(1, 4)
        with torch.no_grad():
            self.weight_hh.set_(weight_hh_data)
        # The bias is just set to zero vectors.
        init.constant(self.bias.data, val=0)
        # Initialization of BN parameters.
        self.bn_ih.reset_parameters()
        self.bn_hh.reset_parameters()
        self.bn_c.reset_parameters()
        self.bn_ih.bias.data.fill_(0)
        self.bn_hh.bias.data.fill_(0)
        self.bn_ih.weight.data.fill_(0.1)
        self.bn_hh.weight.data.fill_(0.1)
        self.bn_c.weight.data.fill_(0.1)

    def forward(self, input_, hx, time):
        """
        Args:
            input_: A (batch, input_size) tensor containing input
                features.
            hx: A tuple (h_0, c_0), which contains the initial hidden
                and cell state, where the size of both states is
                (batch, hidden_size).
            time: The current timestep value, which is used to
                get appropriate running statistics.

        Returns:
            h_1, c_1: Tensors containing the next hidden and cell state.
        """

        h_0, c_0 = hx
        batch_size = h_0.size(0)
        bias_batch = (self.bias.unsqueeze(0)
                      .expand(batch_size, *self.bias.size()))
        wh = torch.mm(h_0, self.weight_hh)
        wi = torch.mm(input_, self.weight_ih)
        bn_wh = self.bn_hh(wh, time=time)
        bn_wi = self.bn_ih(wi, time=time)
        f, i, o, g = torch.split(bn_wh + bn_wi + bias_batch,
             self.hidden_size, dim=1)
        c_1 = torch.sigmoid(f) * c_0 + torch.sigmoid(i) * torch.tanh(g)
        h_1 = torch.sigmoid(o) * torch.tanh(self.bn_c(c_1, time=time))

        return h_1, c_1


class LSTM(nn.Module):

    """A module that runs multiple steps of LSTM."""

    def __init__(
            self, cell_class, input_size, hidden_size, num_layers=1,
            use_bias=True, batch_first=False, dropout=0,
            time_window=32,
            **kwargs
         ):

        super(LSTM, self).__init__()
        self.cell_class = cell_class
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_bias = use_bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.time_window = time_window

        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            cell = cell_class(input_size=layer_input_size,
                              hidden_size=hidden_size,
                              **kwargs)
            setattr(self, 'cell_{}'.format(layer), cell)
        self.dropout_layer = nn.Dropout(dropout)
        self.reset_parameters()

    def get_cell(self, layer):
        return getattr(self, 'cell_{}'.format(layer))

    def reset_parameters(self):
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            cell.reset_parameters()

    @staticmethod
    def _forward_rnn(cell, input_, length, hx):

        T = input_.size(0)
        output = []

        for time in range(T):
            if isinstance(cell, BNLSTMCell):
                h_next, c_next = cell(input_=input_[time], hx=hx, time=time)
            else:
                h_next, c_next = cell(input_=input_[time], hx=hx)

            mask = (time < length).float().unsqueeze(1).expand_as(h_next)
            h_next = h_next*mask + hx[0]*(1 - mask)
            c_next = c_next*mask + hx[1]*(1 - mask)
            hx_next = (h_next, c_next)
            output.append(h_next)
            hx = hx_next

        output = torch.stack(output, 0)

        return output, hx

    def forward(self, input_, length=None, hx=None):

        if self.batch_first:
            input_ = input_.transpose(0, 1)

        T, batch_size, _ = input_.size()

        if length is None:
            length = Variable(torch.LongTensor([self.time_window] * batch_size))
            if input_.is_cuda:
                device = input_.get_device()
                length = length.cuda(device)

        if hx is None:
            # init hidden and cell if not given
            hx = (
                Variable(
                    xavier_uniform(
                        input_.device,
                        self.num_layers, batch_size, self.hidden_size,
                        dtype=input_.dtype
                        )
                ),
                Variable(
                    xavier_uniform(
                        input_.device,
                        self.num_layers, batch_size, self.hidden_size,
                        dtype=input_.dtype
                        )
                )
            )

        h_n = []
        c_n = []
        layer_output = None

        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            hx_layer = (hx[0][layer,:,:], hx[1][layer,:,:])

            if layer == 0:
                layer_output, (layer_h_n, layer_c_n) = LSTM._forward_rnn(
                    cell=cell, input_=input_, length=length, hx=hx_layer)
            else:
                layer_output, (layer_h_n, layer_c_n) = LSTM._forward_rnn(
                    cell=cell, input_=layer_output, length=length, hx=hx_layer)

            input_ = self.dropout_layer(layer_output)
            h_n.append(layer_h_n)
            c_n.append(layer_c_n)

        output = layer_output
        h_n = torch.stack(h_n, 0)
        c_n = torch.stack(c_n, 0)

        return output, (h_n, c_n)

# ------- baselineLSTM 1 --------

class StackedLSTM(nn.Module):
    # to be used as baseline

    # this module has same IO batch shapes as TSTModel:
    # in: batch x time x src_dim
    # out: batch x n_labels

    def __init__(
            self,
            src_dim: int = 128, # dimensionality of data
            dim: int = 10, # dimensionality of model/embedded data
            num_layers: int = 4,
            dropout: float = 0.1,
            emb_dropout: float = 0.1,
            fc_dropout: float = 0.1,
            freeze: bool = False,
            norm_type: str = "nn.BatchNorm1d",
            time_window: int = 24,
            task: str = "pretrain", # regression/pretraining/classification
            n_labels: int = 7,
            **kwargs
        ):
        super().__init__()

        if freeze:
            raise NotImplementedError("freeze")

        norm = eval(norm_type)

        self.num_layers = num_layers

        self.rnn0 = nn.LSTM(
            input_size=src_dim,
            hidden_size=dim,
            num_layers=1,
            batch_first=True,
        )
        # (always add extra dropouts as, according to nn.LSTM docs, last layer is not succeeded by dropout via nn.LSTM dropuot kwarg)
        self.drop0 = nn.Dropout(emb_dropout)

        # Intermediate Stack
        stack = [
            [
                nn.LSTM(
                    input_size=dim,
                    hidden_size=dim,
                    num_layers=1,
                    batch_first=True,
                ),
                nn.Dropout(dropout)
            ] for i in range(1, (num_layers - 1))
        ]
        self.stack = nn.ModuleList(
            [lstm_or_drop for lstm_plus_drop in stack for lstm_or_drop in lstm_plus_drop]
        )

        # Last Stack
        self.rnnLast = nn.LSTM(
            input_size=dim,
            hidden_size=dim, # confirm intermediate layers all use hidden dim
            num_layers=1,
            batch_first=True
        )

        modules = []
        modules.append(nn.Dropout(fc_dropout))
        modules.append(nn.Linear(dim, n_labels))
        modules.append(nn.LogSoftmax(dim=-1))

        self.head = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:

        # print(f"LSTM fwd: x.shape={x.shape}")

        x, _ = self.rnn0(x)
        x = self.drop0(x)

        for i in range(self.num_layers-1):
            if i % 2:
                # dropout layer
                x = self.stack[i](x)
            else:
                # lstm layer
                x, _ = self.stack[i](x)

        out = self.head(x[:,-1,:])

        # print(f"LSTM fwd: out.shape={out.shape}")
        return out

# ------- baselineLSTM 2 --------

class SingleStackLSTM(nn.Module):
    def __init__(
            self,
            src_dim: int = 128, # dimensionality of data
            dim: int = 10, # dimensionality of model/embedded data
            num_layers: int = 4,
            dropout: float = 0.1,
            fc_dropout: float = 0.1,
            freeze: bool = False,
            norm_type: str = "nn.BatchNorm1d",
            time_window: int = 24,
            task: str = "pretrain", # regression/pretraining/classification
            n_labels: int = 7,
            **kwargs
        ):

        super().__init__()

        # lstm
        self.rnn = nn.LSTM(
            input_size=src_dim,
            hidden_size=dim, # confirm intermediate layers all use hidden dim
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        self.task = task = task.lower()

        # construct task head
        head_list = [nn.Dropout(fc_dropout)] if fc_dropout else []

        if "cl" in task:
            # (add extra dropout as, according to nn.LSTM docs, last layer is not succeeded by dropout
            head_list.append(nn.Linear(dim, n_labels))
            head_list.append(nn.LogSoftmax(dim=-1))

        elif "vaeenc" in task:
            # used by VAE encoder
            head_list += [
                nn.Flatten(1),
                nn.Linear(
                    num_layers * 2 * dim,
                    n_labels
                )
            ]
        elif "vaedec" in task:
            # used by VAE decoder
            head_list += [
                nn.Linear(dim, n_labels),
            ]

        self.head = nn.Sequential(*head_list)

        self.num_layers = num_layers
        self.dim = dim

    def forward(self, x: torch.Tensor, mask=None, hx=None) -> torch.Tensor:
        # unroll internally
        intermediate, encoder_states = self.rnn(x, hx=hx)
        if "cl" in self.task or "vaedec" in self.task:
            out = self.head(intermediate[:,-1,:])
        elif "vaeenc" in self.task:
            # VAE enc
            # encoder states is tup(h,c) with h.shape==c.shape==[num_layers, batch, dim]
            head_input = torch.cat([state.transpose(0,1) for state in encoder_states], dim=-1)
            out = self.head(head_input)
        else:
            raise NotImplementedError(f"{task} for SingleStackLSTM")
        return out


class TSLSTM(nn.Module):
    # to be used as comparison to TSTModel

    # this module has same IO batch shapes as TSTModel:
    # in: batch x time x src_dim
    # out: batch x n_labels

    def __init__(
            self,
            src_dim: int = 128, # dimensionality of data
            dim: int = 10, # dimensionality of model/embedded data
            num_layers: int = 4,
            dropout: float = 0.1,
            emb_dropout: float = 0.1,
            fc_dropout: float = 0.1,
            freeze: bool = False,
            norm_type: str = "nn.BatchNorm1d",
            time_window: int = 24,
            task: str = "pretrain", # regression/pretraining/classification
            n_labels: int = 7,
            lstm: str = "batch",
            **kwargs
        ):
        super().__init__()

        if freeze:
            raise NotImplementedError("freeze")

        lstm_cell = BNLSTMCell if "batch" in lstm.lower() else LSTMCell

        # lstm
        self.rnn = LSTM(
            cell_class=lstm_cell,
            input_size=src_dim,
            hidden_size=dim, # confirm intermediate layers all use hidden dim
            max_length=time_window,
            num_layers=num_layers,
            use_bias=True,
            batch_first=True,
            dropout=dropout,
            # **kwargs
        )

        self.task = task = task.lower()

        head_list = [nn.Dropout(fc_dropout)] if fc_dropout else [] # from https://github.com/timeseriesAI/tsai/blob/main/tsai/models/TST.py

        if "pre" in task:
            # pretrain: reconstruct input
            head_list += [
                nn.Linear(dim, src_dim)
            ]

        elif "reg" in task:
            # regression: predict n scalars
            head_list += [
                nn.Flatten(1),
                nn.Linear(dim, n_labels)
            ]
        elif "cl" in task:
            # classification: predict distribution over n labels (Softmax in CrossEntropyLoss)
            head_list += [
                nn.Flatten(1),
                nn.Linear(dim, n_labels),
                nn.LogSoftmax(dim=-1),
            ]
        elif "vaeenc" in task:
            # VAE encoder
            head_list = []
        else:
            raise NotImplementedError(task)

        self.head = nn.Sequential(
            *head_list
        )

    def forward(self, x: torch.Tensor, mask=None, **kwargs) -> torch.Tensor:
        # unroll internally
        intermediate, _ = self.rnn(x, **kwargs)
        out = self.head(intermediate[-1,:,:])
        return out


