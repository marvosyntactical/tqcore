# -*- coding: utf-8 -*-

from typing import Callable, Optional, Dict

from functools import partial

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable

from .quantizable_layer import \
    QuantStub, DeQuantStub, \
    QListener, \
    QAdd, QMul, QMatMul, \
    QSoftmax, \
    QPositionalEncoding, \
    QFill, \
    QReLU6, \
    QLinear, \
    FFT, \
    NonQuantizableModuleWrap, \
    QPlotter, \
    CLIPPING_MODULES, \
    SYMMETRIZING_MODULES
from .batchnorm import QBatchNorm1dTranspose, QBNFoldableTranspose, FPBatchNorm1dTranspose
from .qtensor import QTensor
from .config import QuantStage

linear_cls = QLinear

# NOTE the following modules are copied from tst to avoid this import:
# from tst.transformer import MultiHeadedAttention, MaskedMSE, XentLoss

class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention module from "Attention is All You Need".

    Implementation modified from JoeyNMT.
    https://github.com/joeynmt/joeynmt/blob/master/joeynmt/transformer_layers.py

    Implementation therein modified from OpenNMT-py.
    https://github.com/OpenNMT/OpenNMT-py
    """

    def __init__(self, num_heads: int, dim: int, dropout: float = 0.1):
        """
        Create a multi-headed attention layer.
        :param num_heads: the number of heads
        :param dim: model dim (must be divisible by num_heads)
        :param dropout: probability of dropping a unit
        """
        super().__init__()

        assert dim % num_heads == 0

        self.head_size = head_size = dim // num_heads
        self.model_size = dim
        self.num_heads = num_heads

        self.k_layer = nn.Linear(dim, num_heads * head_size)
        self.v_layer = nn.Linear(dim, num_heads * head_size)
        self.q_layer = nn.Linear(dim, num_heads * head_size)

        self.output_layer = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, k: Tensor, v: Tensor, q: Tensor, mask: Optional[Tensor] = None):
        """
        Computes multi-headed attention.

        :param k: keys   [B, W, D] with W being the time window.
        :param v: values [B, W, D]
        :param q: query  [B, W, D]
        :param mask: optional mask [B, 1, W]
        :return:
        """
        batch_size = k.size(0)
        num_heads = self.num_heads

        # project the queries (q), keys (k), and values (v)
        k = self.k_layer(k)
        v = self.v_layer(v)
        q = self.q_layer(q)

        # reshape q, k, v for our computation to (batch_size, num_heads, ..., ...)
        k = k.view(batch_size, num_heads, self.head_size, -1)
        v = v.view(batch_size, num_heads, -1, self.head_size)
        q = q.view(batch_size, num_heads, -1, self.head_size)

        # compute scores
        q = q / math.sqrt(self.head_size)

        # batch x num_heads x query_len x key_len
        scores = q @ k

        # apply the mask (if we have one)
        # we add a dimension for the heads to it below: [B, 1, 1, W]
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(mask==torch.as_tensor(False), float('-inf'))
            # scores = scores.masked_fill(~mask, -1e10)

        # DONT apply attention dropout and compute context vectors.
        attention = self.softmax(scores)
        attention = self.dropout(attention)

        # get context vector (select values with attention) and reshape
        # back to [B, W, D]
        context = attention @ v
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, num_heads * self.head_size
        )

        output = self.output_layer(context)

        return output

# this file contains quantized versions of layers used in tst_pytorch/modules.py

class MaskedMSE(nn.Module):
    """
    from stackoverflow:
    https://discuss.pytorch.org/t/how-to-write-a-loss-function-with-mask/53461/
    """
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, pred, trgt, mask):
        # mask is
        # 1 where loss should be considered,
        # 0 where it should be dropped
        return (((pred-trgt)*mask)**2).sum() / mask.sum()

class XentLoss(nn.Module):
    """
    taken from: https://github.com/joeynmt/joeynmt/blob/master/joeynmt/loss.py
    Cross-Entropy Loss with optional label smoothing
    """

    def __init__(self, n_labels: int, smoothing: float = 0.0):
        super().__init__()

        self.smoothing = smoothing
        self.n_labels = n_labels

        if self.smoothing <= 0.0:
            # standard xent loss
            self.criterion = nn.NLLLoss(reduction='sum')
        else:
            # custom label-smoothed loss, computed with KL divergence loss
            self.criterion = nn.KLDivLoss(reduction='sum')

    def _smooth_targets(self, targets: Tensor, n_labels: int = None):
       """
       Smooth target distribution. All non-reference words get uniform
       probability mass according to "smoothing".
       :param targets: target indices, batch
       :param n_labels:
       :return: smoothed target distributions, batch x n_labels
       """
       if n_labels is None:
           n_labels = self.n_labels

       # batch x n_labels
       smooth_dist = targets.new_zeros((targets.size(0), n_labels)).float()
       # fill distribution uniformly with smoothing
       smooth_dist.fill_(self.smoothing / (n_labels - 2))
       # assign true label the probability of 1-smoothing ("confidence")
       smooth_dist.scatter_(1, targets.unsqueeze(1).data, 1.0-self.smoothing)

       return Variable(smooth_dist, requires_grad=False)

    def forward(self, outputs, targets):
        """
        Compute the cross-entropy between logits and targets.
        If label smoothing is used, target distributions are not one-hot, but
        "1-smoothing" for the correct target token and the rest of the
        probability mass is uniformly spread across the other tokens.
        :param outputs: values as predicted by model
        :param targets: target indices
        :return:
        """
        log_probs = F.log_softmax(outputs)

        if self.smoothing > 0:
            targets = self._smooth_targets(
                targets=targets.contiguous().view(-1),
                n_labels=log_probs.size(-1))
            # targets: distributions with batch x n_labels
            assert log_probs.contiguous().view(-1, log_probs.size(-1)).shape \
                == targets.shape
        else:
            # targets: indices with batch
            targets = targets.contiguous().view(-1)
        loss = self.criterion(
                log_probs.contiguous().view(-1, log_probs.size(-1)), targets)
        return loss

class QMultiHeadedAttention(nn.Module):
    """
    Q Multi-Head Attention module from "Attention is All You Need"

    Implementation modified from OpenNMT-py.
    https://github.com/OpenNMT/OpenNMT-py

    made quantizable, taken from
    https://github.com/joeynmt/joeynmt/blob/master/joeynmt/transformer_layers.py
    """

    def __init__(self, num_heads: int, dim: int, dropout: float = 0.1, layer_num: int = 0, **qkwargs):
        """
        Create a multi-headed attention layer.
        :param num_heads: the number of heads
        :param dim: model dim (must be divisible by num_heads)
        :param dropout: probability of dropping a unit
        """
        super().__init__()

        assert dim % num_heads == 0

        pn = lambda s: s + " " + str(layer_num)

        self.head_size = head_size = dim // num_heads
        self.model_size = dim
        self.num_heads = num_heads

        self.k_layer = linear_cls(dim, num_heads * head_size, qkwargs=qkwargs)
        self.post_kl = QListener(self.k_layer, plot_name=pn("k_layer"), **qkwargs)

        self.v_layer = linear_cls(dim, num_heads * head_size, qkwargs=qkwargs)
        self.post_vl = QListener(self.v_layer, plot_name=pn("v_layer"), **qkwargs)

        self.q_layer = linear_cls(dim, num_heads * head_size, qkwargs=qkwargs)
        self.post_ql = QListener(self.q_layer, plot_name=pn("q_layer"), **qkwargs)

        scale = 1./head_size**6
        # scale = 1./math.sqrt(self.head_size)
        # assert False, "remember to correct scale"

        self.qkMatMul = QMatMul(factor=scale, **qkwargs)
        self.qkl = QListener(self.qkMatMul, plot_name=pn("qk matmul"),**qkwargs)

        self.qMask = QFill(**qkwargs)
        self.qMaskl = QListener(self.qMask, dont_fakeQ=True, plot_name=pn("qmask"), **qkwargs)

        qsoft = qkwargs["transformer"]["qsoftmax"]
        if qsoft:
            self.qsoftmax = QSoftmax(dim=-1, layer_num=layer_num, **qkwargs)
        else:
            self.qsoftmax = NonQuantizableModuleWrap(
                nn.Softmax(dim=-1), **qkwargs
            )
        self.dropout = nn.Dropout(dropout)

        self.avMatMul = QMatMul(**qkwargs)
        self.avl = QListener(self.avMatMul, plot_name=pn("av matmul"), **qkwargs)

        self.output_layer = linear_cls(dim, dim, qkwargs=qkwargs)
        self.plot_step_counter = 0


    def forward(self, k: Tensor, v: Tensor, q: Tensor, mask: Tensor = None):
        """
        Computes multi-headed attention.

        :param k: keys   [B, W, D] with W being the time window.
        :param v: values [B, W, D]
        :param q: query  [B, W, D]
        :param mask: optional mask [B, 1, W]
        :return:
        """
        batch_size = k.shape[0]
        num_heads = self.num_heads

        # project the queries (q), keys (k), and values (v)
        k = self.k_layer(k)
        k = self.post_kl(k)

        v = self.v_layer(v)
        v = self.post_vl(v)

        q = self.q_layer(q)
        q = self.post_ql(q)

        # reshape q, k, v for our computation to (batch_size, num_heads, ..., ...)
        k = k.view(batch_size, num_heads, self.head_size, -1)
        v = v.view(batch_size, num_heads, -1, self.head_size)
        q = q.view(batch_size, num_heads, -1, self.head_size)

        # compute scores
        # batch x num_heads x query_len x key_len
        scores = self.qkMatMul(q, k)
        # input(f"Scores stats (please be smol in magnitude): {scores.min().item()}, {scores.max().item()}")
        scores = self.qkl(scores)

        # apply the mask (if we have one)
        # we add a dimension for the heads to it below: [B, 1, 1, W]
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
            scores = self.qMask(scores, mask)
            print(f"Got mask={mask!=0}")
            print(f"qMaskl was activated! qMaskl.__stats__: {self.qMaskl.__stats__}")
            self.qMaskl(scores)

        # # normalize context vectors.
        attention = self.qsoftmax(scores)
        # attention = scores
        attention = self.dropout(attention)

        # get context vector (select values with attention) and reshape
        # back to [B, W, D]
        context = self.avMatMul(attention, v)
        context = self.avl(context)

        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, num_heads * self.head_size)

        output = self.output_layer(context)
        self.plot_step_counter += 1

        return output


class QPositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-forward layer
    Projects to ff_size and then back down to input_size.
    """

    def __init__(self, input_size, ff_size, time_window, dropout=0.1, activ: str="nn.ReLU6", layer_num: int =0, **qkwargs):
        """
        Initializes position-wise feed-forward layer.
        :param input_size: dimensionality of the input.
        :param ff_size: dimensionality of intermediate representation
        :param dropout:
        """
        super().__init__()
        activation = eval(activ.strip())(**qkwargs)
        print(f"Initiated QFeedFwd with activation={activation}")
        pn = lambda s: s + " " + str(layer_num)

        modules = [
            linear_cls(input_size, ff_size, qkwargs=qkwargs),
            activation,
        ]
        modules += [
            QListener(*modules, clipped_distribution=True, plot_name=pn("pwff "+activ), **qkwargs),
            linear_cls(ff_size, input_size, qkwargs=qkwargs)
        ]
        modules += [
            QListener(modules[-1], plot_name=pn("pwff out"), **qkwargs),
        ]
        self.pwff_layer = nn.Sequential(*modules)

    def forward(self, x):
        out = self.pwff_layer(x)
        return out


class QTransformerEncoderLayer(nn.Module):
    """
    One Transformer encoder layer has a Multi-head attention layer plus
    a position-wise feed-forward layer.

    From https://github.com/joeynmt/joeynmt/blob/master/joeynmt/transformer_layers.py
    """

    def __init__(self,
             dim: int = 0,
             ff_size: int = 0,
             num_heads: int = 0,
             dropout: float = 0.1,
             time_window: int = 50,
             bn_mom: float = 0.9,
             activ: str = "QReLU6",
             fft: bool = False,
             qkwargs: Dict = None,
             layer_num: int = 0,
             **kwargs
        ):
        """
        A single quantizable Transformer layer.
        :param dim:
        :param ff_size:
        :param num_heads:
        :param dropout:
        :param qkwargs: keyword args for QListener
        """
        super().__init__()

        # for unique plotnames:
        pn = lambda s: s + " " + str(layer_num)

        self.fft = fft

        # NOTE DEBUG TODO:
        # add these as cfg params
        # (preferrably dont leave them out entirely once NonQuant works)
        # => mix (NonQuant) quantization has priority
        qtransf_kwargs = qkwargs["transformer"]

        self.has_mix = qtransf_kwargs["has_mix"]
        self.qattn = qtransf_kwargs["qattn"]
        self.has_res1 = qtransf_kwargs["has_res1"]
        self.has_bn = qtransf_kwargs["has_bn"]
        self.qbn= qtransf_kwargs["qbn"]
        self.has_res2 = qtransf_kwargs["has_res2"]

        # self.simulate_folding = 0 # NOTE deprecated
        # if self.simulate_folding:
        #     BatchNormMod = QBNFoldableTranspose

        if self.qbn:
            BatchNormMod = QBatchNorm1dTranspose
        else:
            def wrapped_bn(dim, momentum=None, qkwargs=None, **kwargs):
                return NonQuantizableModuleWrap(
                    FPBatchNorm1dTranspose(
                        dim, momentum=momentum, **kwargs
                    ), **qkwargs
                )
            BatchNormMod = wrapped_bn

        mix_output_layer = []
        if self.has_mix:
            if not self.fft:
                if self.qattn:
                    self.mixer = QMultiHeadedAttention(
                        num_heads, dim,
                        dropout=dropout,
                        layer_num=layer_num, **qkwargs)
                    mix_output_layer = [self.mixer.output_layer]
                else:
                    self.mixer = NonQuantizableModuleWrap(
                        MultiHeadedAttention(
                            num_heads, dim, dropout=dropout
                        ), plot_name=pn("fp softmax"), **qkwargs
                    )

                    mix_output_layer = [] # layer that should be updated by next listener
                # mix_output_layer = [self.mixer.fp_module]
                self.mix = lambda x, mask: self.mixer(x,x,x,mask)
            else:
                self.mixer = NonQuantizableModuleWrap(FFT(), **qkwargs)
                mix_output_layer = []
                self.mix = lambda x, mask: self.mixer(x, mask)

        if self.has_res1:
            self.dropout1 = nn.Dropout(dropout)
            self.add1 = QAdd(**qkwargs)
            self.add1l = QListener(
                * [self.add1] + mix_output_layer,
                clipped_distr=False,
                plot_name=pn("residual 1"),
                **qkwargs,
            )
        if self.has_bn:
            self.norm1 = BatchNormMod(dim, momentum=bn_mom, qkwargs=qkwargs, **kwargs)
            if not self.has_res2:
                self.norm1l = QListener(self.norm1, plot_name=pn("norm 1"), **qkwargs)

        self.feed_forward = QPositionwiseFeedForward(
            dim,
            ff_size=ff_size,
            dropout=dropout,
            time_window=time_window,
            activ=activ,
            layer_num=layer_num,
            **qkwargs
        )

        if self.has_res2:
            assert self.has_bn
            self.dropout2 = nn.Dropout(dropout)
            self.add2 = QAdd(**qkwargs)
            self.add2l = QListener(
                self.add2,
                self.feed_forward.pwff_layer._modules[str(len(self.feed_forward.pwff_layer._modules)-1)],
                self.norm1,
                clipped_distr=False,
                plot_name=pn("residual 2"),
                **qkwargs
            )
        if self.has_bn:
            self.norm2 = BatchNormMod(dim, momentum=bn_mom, qkwargs=qkwargs, **kwargs)
            self.norm2l = QListener(self.norm2, plot_name=pn("norm 2"), **qkwargs)

        self.plot_step_counter = 0

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Forward pass for a single transformer encoder layer.
        First applies layer norm, then self attention,
        then dropout with residual connection (adding the input to the result),
        and then a position-wise feed-forward layer.

        :param x: layer input
        :param mask: input mask
        :return: output tensor
        """
        # if isinstance(x, QTensor):
        #     input(f"In layer at stage={self.mixer.stage}, input is quantized={x.quantized}, mask={type(mask)}.")

        if self.has_mix:
            h = self.mix(x, mask)
        else:
            h = x

        if self.has_res1:
            res1 = self.add1(h, self.dropout1(x))
            res1 = self.add1l(res1)
        else:
            res1 = h

        if self.has_bn:
            h = self.norm1(res1)
            if not self.has_res2:
                h = self.norm1l(h)
        else:
            h = res1

        ff_out = self.feed_forward(h)

        if self.has_res2:
            res2 = self.add2(ff_out, self.dropout2(h))
            res2 = self.add2l(res2)
        else:
            res2 = ff_out

        if self.has_bn:
            o = self.norm2(res2)
            o = self.norm2l(o)
        else:
            o = res2

        self.plot_step_counter += 1
        return o


class QTransformerEncoder(nn.Module):
    """
    Transformer Encoder.
    From https://github.com/joeynmt/joeynmt/blob/master/joeynmt/encoders.py
    """

    def __init__(
             self,
             src_dim: int = 187,
             dim: int = 512,
             ff_size: int = 2048,
             num_layers: int = 8,
             num_heads: int = 4,
             dropout: float = 0.1,
             emb_dropout: float = 0.1,
             freeze: bool = False,
             time_window: int = 24,
             activ: str = "nn.ReLU6",
             fft: bool = False,
             qkwargs: Dict = None
        ):
        """
        Initializes the Transformer. NOTE: you still have to set self.mu, self.sigma later!
        :param src_dim: dimensionality of data
        :param dim: hidden size and size of embeddings
        :param ff_size: position-wise feed-forward layer size.
          (Typically this is 2*dim)
        :param num_layers: number of layers
        :param num_heads: number of heads for multi-headed attention
        :param dropout: dropout probability for Transformer layers
        :param emb_dropout: Is applied to the input (word embeddings).
        :param freeze: freeze the parameters of the encoder during training
        :param qkwargs:
        """
        super().__init__()

        self.quantStub = QuantStub(**qkwargs)
        self.input_listener = QListener(self.quantStub, calibration="minmax", plot_name="input", **qkwargs)

        self.embedding = linear_cls(src_dim, dim, qkwargs=qkwargs)
        self.emb_listener = QListener(self.embedding, plot_name="embedding", **qkwargs)

        # TODO FIXME add these again
        self.has_pe = qkwargs["transformer"]["has_pe"]
        self.qpe = qkwargs["transformer"]["qpe"]
        if self.has_pe:
            if not self.qpe:
                self.pe = NonQuantizableModuleWrap(
                    QPositionalEncoding(dim, time_window)
                    ,**qkwargs
                )
            else:
                self.pe = QPositionalEncoding(dim, time_window)
                self.pe_listener = QListener(self.pe, plot_name="pos enc", **qkwargs)

        self.emb_dropout = nn.Dropout(p=emb_dropout)

        # build all (num_layers) layers
        self.layers = nn.ModuleList([
            QTransformerEncoderLayer(
                dim=dim, ff_size=ff_size,
                num_heads=num_heads, dropout=dropout,
                time_window=time_window, activ=activ,
                fft=fft, qkwargs=qkwargs,
                layer_num=l # for unique plot names
            )
            for l in range(num_layers)
        ])
        # for i in range(self.layers):
        #     setattr(self.layers["{i}"], "i", i)

        if freeze:
            raise NotImplementedError("TODO Implement Freezing everything but head as in TST paper")

        self.plot_step_counter = 0

    def forward(self,
                x: Tensor,
                mask: Optional[Tensor] = None,
                **kwargs) -> (Tensor, Tensor):
        """
        Pass the input (and mask) through each layer in turn.
        Applies a Transformer encoder to sequence of embeddings x.
        The input mini-batch x needs to be sorted by src length.
        x and mask should have the same dimensions [batch, time, dim].

        :param x: non-embedded src inputs,
            shape (batch_size, src_len, embed_size)
        :param mask: indicates padding areas (zeros where padding), shape
            (batch_size, 1, src_len)
        :return:
            - output: hidden states with
                shape (batch_size, max_length, directions*hidden),
            - hidden_concat: last hidden state with
                shape (batch_size, directions*hidden)
        """
        self.plot_step_counter += 1

        # data normalization happens in data.py
        # x = (x - x.mean()) / x.std()

        x = self.quantStub(x)
        x = self.input_listener(x)

        # self_is_quant = self.layers._modules["0"].norm1.stage == QuantStage.Quantized
        # if self_is_quant:
        #     assert x.quantized, self.input_listener.forward

        # print_qt_stats("input", x, stage=self.quantStub.stage, step=self.plot_step_counter, plot="always")
        x = self.embedding(x)
        x = self.emb_listener(x)

        # if self_is_quant:
        #     assert x.quantized

        if self.has_pe:
            x = self.pe(x) # add position encoding to word embeddings
            if self.qpe:
                x = self.pe_listener(x)

        x = self.emb_dropout(x)

        # if self.layers._modules["0"].norm1.stage == QuantStage.Quantized:
        #     assert x.quantized
        #     assert self.quantStub.stage == QuantStage.Quantized
        #     assert self.layers._modules["0"].mixer.stage == QuantStage.Quantized

        if self.quantStub.stage == QuantStage.Quantized:
            assert x.quantized
            assert not isinstance(mask, Tensor)
        elif self.quantStub.stage == QuantStage.QAT:
            pass
            # print(f"{x}\nquantized?")

        for i, layer in enumerate(self.layers):
            x = layer(x, mask)

        return x

    def __repr__(self):
        if not self.layers[0].fft:
            pass
            # TODO uncomment after removing has_mix/has_bn/has_res .... # FIXME
            # s = "%s(num_layers=%r, num_heads=%r)" % (
            #     self.__class__.__name__, len(self.layers),
            #     self.layers[0].mixer.fp_module.num_heads,
            #     self.layers[0].mixer.num_heads,
            # )
            s="f{self.__class__.__name__}:(FIXME display info)"
        else:
            s = "%s(num_layers=%r, fft=True)" % (
                self.__class__.__name__, len(self.layers)
            )
        return s

class QTSTModel(nn.Module):
    """
    Quantizable Classification/regression with Time Series Transformer.
    Structure from https://github.com/dhlee347/pytorchic-bert/blob/master/classify.py """
    def __init__(
            self,
            src_dim: int = 128, # dimensionality of data
            dim: int = 128, # dimensionality of model/embedded data
            ff_size: int = 256,
            num_layers: int = 3,
            num_heads: int = 8,
            dropout: float = 0.1,
            emb_dropout: float = 0.1,
            freeze: bool = False,
            activ: str = "nn.ReLU6",
            encoder: QTransformerEncoder = None,
            time_window: int = 24,
            task: str = "pretrain", # regression/pretraining/classification
            n_labels: int = 10, # classification only
            fc_dropout: float = 0.0,
            fft: bool = False,
            qkwargs: Dict = None,
            **kwargs # settings for quantizable modules
        ):
        super().__init__()

        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = QTransformerEncoder(
                src_dim=src_dim,
                dim=dim,
                ff_size=ff_size,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
                emb_dropout=emb_dropout,
                freeze=freeze,
                time_window=time_window,
                activ=activ,
                fft=fft,
                qkwargs=qkwargs,
            )

        self.task = task = task.lower()

        head = [nn.Dropout(fc_dropout)] if fc_dropout else [] # from https://github.com/timeseriesAI/tsai/blob/main/tsai/models/TST.py

        if "pre" in task:
            # pretrain: reconstruct input
            head += [
                linear_cls(dim, src_dim, qkwargs=qkwargs),
            ]
            head += [
                QListener(head[-1], plot_name="head", **qkwargs),
                DeQuantStub(**qkwargs),
            ]

        elif "reg" in task:
            # regression: predict n scalars
            head += [
                nn.Flatten(1),
                linear_cls(time_window * dim, n_labels, qkwargs=qkwargs),
            ]
            head += [
                QListener(head[-1], plot_name="head", **qkwargs),
                DeQuantStub(**qkwargs),
            ]
        elif "cl" in task:
            # classification: predict distribution over n labels (Softmax in CrossEntropyLoss)
            head += [
                nn.Flatten(1),
                linear_cls(time_window * dim, n_labels, qkwargs=qkwargs),
            ]
            head += [
                QListener(head[-1], plot_name="head", **qkwargs),
                DeQuantStub(**qkwargs),
                nn.LogSoftmax(dim=-1),
            ]
        else:
            raise NotImplementedError(task)

        self.head = nn.Sequential(
            *head
        )

    def forward(self, src, mask=None):

        h = self.encoder(src, mask)
        out = self.head(h)

        return out

SYMMETRIZING_MODULES += [
    QPositionalEncoding,
]


