# -*- coding: utf-8 -*-

from typing import Callable, Optional, Dict, List

from functools import partial
from copy import deepcopy

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable

from .quantizable_layer import \
    QuantizableModule, \
    QuantStub, DeQuantStub, \
    QListener, \
    QAdd, QMul, QMatMul, \
    QSoftmax, \
    QPositionalEncoding, \
    QFill, \
    QCat, \
    QLabel, \
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

        self.k_layer = QLinear(dim, num_heads * head_size, qkwargs=qkwargs)
        self.post_kl = QListener(self.k_layer, plot_name=pn("k_layer"), **qkwargs)

        self.v_layer = QLinear(dim, num_heads * head_size, qkwargs=qkwargs)
        self.post_vl = QListener(self.v_layer, plot_name=pn("v_layer"), **qkwargs)

        self.q_layer = QLinear(dim, num_heads * head_size, qkwargs=qkwargs)
        self.post_ql = QListener(self.q_layer, plot_name=pn("q_layer"), **qkwargs)

        # scale = head_size**-4
        scale = 1./math.sqrt(self.head_size)

        self.qkMatMul = QMatMul(factor=scale, **qkwargs)
        self.qkl = QListener(self.qkMatMul, plot_name=pn("qk matmul"),**qkwargs)

        self.qMask = QFill(**qkwargs)
        self.qMaskl = QListener(self.qMask, dont_fakeQ=True, plot_name=pn("qmask"), **qkwargs)

        qsoft = qkwargs["transformer"]["qsoftmax"]
        if qsoft:
            self.softmax = QSoftmax(
                dim=-1, layer_num=layer_num, **qkwargs
            )
        else:
            self.softmax = NonQuantizableModuleWrap(
                nn.Softmax(dim=-1), **qkwargs
            )
        self.dropout = nn.Dropout(dropout)

        self.avMatMul = QMatMul(**qkwargs)
        self.avl = QListener(self.avMatMul, plot_name=pn("av matmul"), **qkwargs)

        self.output_layer = QLinear(dim, dim, qkwargs=qkwargs)
        self.output_listener = QListener(self.output_layer, plot_name=pn("attn linear out"), **qkwargs)

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
        scores = self.qkl(scores)

        # apply the mask (if we have one)
        # we add a dimension for the heads to it below: [B, 1, 1, W]
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
            scores = self.qMask(scores, mask)
            scores = self.qMaskl(scores)

        # # normalize context vectors.
        attention = self.softmax(scores)
        attention = self.dropout(attention)

        # get context vector (select values with attention)
        context = self.avMatMul(attention, v)
        context = self.avl(context)

        # reshape back to [B, T, D]
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, num_heads * self.head_size
        )
        output = self.output_layer(context)
        output = self.output_listener(output)

        return output


class QPositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-forward layer
    Projects to ff_size and then back down to input_size.
    """

    def __init__(self,
            input_size,
            ff_size,
            time_window,
            dropout=0.1,
            activ: str="nn.ReLU6",
            layer_num: int =0,
            has_output_layer: bool=True,
            has_external_listener: bool=False,
            **qkwargs
        ):
        """
        Initializes position-wise feed-forward layer.
        :param input_size: dimensionality of the input.
        :param ff_size: dimensionality of intermediate representation
        :param dropout:
        """
        super().__init__()
        self.has_output_layer = has_output_layer
        activation = eval(activ.strip())(**qkwargs)
        pn = lambda s: s + " " + str(layer_num)

        modules = [
            QLinear(input_size, ff_size, qkwargs=qkwargs),
            activation,
            nn.Dropout(dropout)
        ]
        modules += [
            QListener(*modules[:2], clipped_distribution=True, plot_name=pn("pwff "+activ), **qkwargs),
        ]
        if self.has_output_layer:
            # output layer precedes a batch norm; bias is redundant in this case
            modules += [
                QLinear(ff_size, input_size, bias=False, qkwargs=qkwargs),
                nn.Dropout(dropout),
            ]
            if not has_external_listener:
                modules += [
                    QListener(modules[-2], plot_name=pn("pwff out"), **qkwargs),
                ]
        else:
            assert not has_external_listener

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
             bn_mom: float = 0.1,
             activ: str = "QReLU6",
             fft: bool = False,
             qkwargs: Dict = None,
             layer_num: int = 0,
             total_num_layers: int = 0,
             skip_layer: Optional[List[QuantizableModule]] = None,
             **kwargs
        ):
        """
        A single quantizable Transformer layer with
        lots of configuration options for ablation studies.

        :param dim: hidden dimensionality of the transformer
        :param ff_size: size of the expanded feed forward layer
        :param dropout: 0.0 < dropout < 1.0 node dropout probability
        :param time_window: length of the sequence
        :param bn_mom: momentum of the batch norm EMA
        :param activ: activation module name string that is evaluated (TODO NOTE UNSAFE)
        :param fft: whether to use a fast fourier transform for time step mixing
        :param layer_num: number of the encoder layer:
        :param total_num_layers: total number of encoder layers:
        :param skip_layer: list containing quantizable layer that will be connected to this layer's first residual
        :param qkwargs: keyword args for QuantizableModules
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

        # in case of key error, please specify these in quantization config transformer section
        self.has_mix = qtransf_kwargs["has_mix"]
        self.qattn = qtransf_kwargs["qattn"]
        self.has_res = qtransf_kwargs["has_res"]
        self.has_bn = qtransf_kwargs["has_bn"]
        self.bn_after_res = qtransf_kwargs["bn_after_res"]
        self.qbn = qtransf_kwargs["qbn"]
        self.fold_bn = qtransf_kwargs["fold_bn"]

        if self.has_mix:
            if not self.fft:
                if self.qattn:
                    self.mixer = QMultiHeadedAttention(
                        num_heads, dim,
                        dropout=dropout,
                        layer_num=layer_num, **qkwargs
                    )
                else:
                    self.mixer = NonQuantizableModuleWrap(
                        MultiHeadedAttention(
                            num_heads, dim, dropout=dropout
                        ), plot_name=pn("fp softmax"), **qkwargs
                    )

                self.mix = lambda x, mask: self.mixer(x,x,x,mask)
            else:
                self.mixer = NonQuantizableModuleWrap(FFT(), **qkwargs)
                self.mix = lambda x, mask: self.mixer(x, mask)

            self.dropout = nn.Dropout(dropout)

        # if bn after res is true in transformer config,
        # feed forward output layer is listened to by the residual2 listener
        ff_listened_to_by_residual2 = (not self.fold_bn) and (self.has_res) and (layer_num < total_num_layers-1) and (self.bn_after_res)

        self.feed_forward = QPositionwiseFeedForward(
            dim,
            ff_size=ff_size,
            dropout=dropout,
            time_window=time_window,
            activ=activ,
            layer_num=layer_num,
            has_output_layer=not self.fold_bn,
            has_external_listener=ff_listened_to_by_residual2,
            **qkwargs
        )

        # residual inputs are either batchnorm, if transformer config has "bn_after_res" set to true;
        # or whatever else comes before: in the case of
        # - residual2, thats just the last layer of the feedforward sequential module
        # - residual1, thats the last layer of the previous transformer layer, or the last
        # encoder layer before the first transformer layer, respectively
        residual1input = []
        residual2input = []

        if self.has_bn:
            if self.fold_bn:
                # this is practically ignored, but make sure the opposite isnt wanted:
                assert self.qbn
                # must be after feed forward layer to fold
                BatchNormMod = QBNFoldableTranspose
            elif self.qbn:
                BatchNormMod = QBatchNorm1dTranspose
            else:
                def wrapped_bn(dim, qkwargs=None, **kwargs):
                    return NonQuantizableModuleWrap(
                        FPBatchNorm1dTranspose(
                            dim, **kwargs
                        ), **qkwargs
                    )
                BatchNormMod = wrapped_bn

            bn_args = deepcopy(kwargs)
            if self.fold_bn:
                # bias is redundant right before a batch norm
                bn_args["linear"] = QLinear(ff_size, dim, bias=False, qkwargs=qkwargs)
                bn_args["dropout"] = dropout

            self.norm1 = BatchNormMod(dim, momentum=bn_mom, qkwargs=qkwargs, **bn_args)
            self.norm1l = QListener(self.norm1, plot_name=pn("norm1"), **qkwargs)

            if self.bn_after_res:
                residual1input += skip_layer
            else:
                residual1input += [self.norm1]

            self.norm2 = BatchNormMod(dim, momentum=bn_mom, qkwargs=qkwargs, **bn_args)
            self.norm2l = QListener(self.norm2, plot_name=pn("norm2"), **qkwargs)

            if self.bn_after_res:
                residual2input += [self.feed_forward.pwff_layer[-2]]
            else:
                residual2input += [self.norm2]

        if self.has_res:
            self.residual1 = QAdd(**qkwargs)
            self.residual1l = QListener(
                * [self.residual1] + residual1input, # TODO NOTE
                clipped_distr=False,
                plot_name=pn("residual1"),
                **qkwargs,
            )

            self.residual2 = QAdd(**qkwargs)
            self.residual2l = QListener(
                * [self.residual2] + residual2input,
                clipped_distr=False,
                plot_name=pn("residual2"),
                **qkwargs,
            )

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Forward pass for a single transformer encoder layer.

        :param x: layer input
        :param mask: input mask
        :return: output tensor
        """

        if self.has_mix:
            h = self.mix(x, mask)
            h = self.dropout(h)
        else:
            h = x

        # BN before residual version:
        if self.has_bn and not self.bn_after_res:
            h = self.norm1(h)
            h = self.norm1l(h)

        if self.has_res:
            h = self.residual1(h, x)
            h = self.residual1l(h)

        # BN after residual version:
        if self.has_bn and self.bn_after_res:
            h = self.norm1(h)
            h = self.norm1l(h)

        ff_out = self.feed_forward(h)

        # BN before residual version:
        if self.has_bn and not self.bn_after_res:
            ff_out = self.norm2(ff_out)
            ff_out = self.norm2l(ff_out)

        if self.has_res:
            out = self.residual2(ff_out, h)
            out = self.residual2l(out)
        else:
            out = ff_out

        # BN after residual version:
        if self.has_bn and self.bn_after_res:
            out = self.norm2(out)
            out = self.norm2l(out)

        return out

    def __str__(self):
        s = "QTransformerEncoderLayer(\n"
        if self.has_mix:
            s += f"\t(mix): {self.mix}\n"
        s += f"\t(ff): {self.feed_forward}\n"
        if self.has_res:
            s += f"\t(res1): {self.residual1}\n"
        if self.has_bn:
            s += f"\t(bn1): {self.norm1}\n"
        if self.has_res:
            s += f"\t(res2): {self.residual2}\n"
        if self.has_bn:
            s += f"\t(bn2): {self.norm2}\n"
        s += ")"
        return s


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
             time_window: int = 24,
             activ: str = "nn.ReLU6",
             fft: bool = False,
             qkwargs: Dict = None,
             learnable_label: bool = False,
        ):
        """
        Quantizable Transformer Encoder.

        :param dim: hidden dimensionality of the transformer
        :param ff_size: size of the expanded feed forward layer
        :param dropout: 0.0 < dropout < 1.0 node dropout probability
        :param time_window: length of the sequence
        :param bn_mom: momentum of the batch norm EMA
        :param activ: activation module name string that is evaluated (TODO NOTE UNSAFE)
        :param fft: whether to use a fast fourier transform for time step mixing
        :param num_layers: total number of encoder layers:
        :param src_dim: dimensionality of data
        :param dim: hidden size and size of embeddings
        :param num_heads: number of heads for multi-headed attention
        :param dropout: dropout probability for Transformer layers
        :param emb_dropout: Is applied to the input (token embedding).
        :param qkwargs:
        """
        super().__init__()


        self.quantStub = QuantStub(**qkwargs)
        self.input_listener = QListener(self.quantStub, calibration="minmax", plot_name="input", **qkwargs)

        self.embedding = QLinear(src_dim, dim, qkwargs=qkwargs)
        self.emb_listener = QListener(self.embedding, plot_name="embedding", **qkwargs)
        self.emb_dropout = nn.Dropout(p=emb_dropout)

        self.learnable_label = learnable_label
        if learnable_label:
            # as in keyword transformer: https://arxiv.org/pdf/2104.00769 Section 3.1
            self.add_label = QLabel(rank=3, cat_dim=1, hidden_dim=2, hidden_size=dim, plot_name="label", **qkwargs)
            time_window += 1

        self.has_pe = qkwargs["transformer"]["has_pe"]
        self.qpe = qkwargs["transformer"]["qpe"]

        if self.has_pe:
            if not self.qpe:
                self.pe = NonQuantizableModuleWrap(
                    QPositionalEncoding(dim, time_window)
                    ,**qkwargs
                )
            else:
                self.pe = QPositionalEncoding(dim, time_window, **qkwargs)
                self.pe_listener = QListener(self.pe, plot_name="pos enc", **qkwargs)

        # build all (num_layers) layers
        layers = []
        for l in range(num_layers):
            # residual comes from previous feedforward in each layer if l > 0
            # otherwise from the last module above
            # (need this information to update the module appropriately using
            # the QListener of the encoder layer's first residual)
            if l > 0:
                skip_layer = layers[-1].residual2 if qkwargs["transformer"]["has_res"] else None
            else:
                skip_layer = self.pe if self.has_pe else (
                    self.add_label if self.learnable_label else (
                        self.embedding
                    )
                )
            layer = QTransformerEncoderLayer(
                dim=dim,
                ff_size=ff_size,
                num_heads=num_heads,
                dropout=dropout,
                time_window=time_window,
                activ=activ,
                fft=fft,
                qkwargs=qkwargs,
                layer_num=l, # for unique plot names
                total_num_layers=num_layers,
                skip_layer=[skip_layer]
            )
            layers += [layer]

        self.layers = nn.ModuleList(layers)

    def forward(
            self,
            x: Tensor,
            mask: Optional[Tensor] = None,
            **kwargs
        ) -> (Tensor, Tensor):
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

        # data normalization by train set statistics already happens in data.py
        # renormalizing per batch turns out to help
        x = (x - x.mean()) / x.std()

        x = self.quantStub(x)
        x = self.input_listener(x)

        x = self.embedding(x)
        x = self.emb_listener(x)
        x = self.emb_dropout(x)

        if self.learnable_label:
            x = self.add_label(x)

        if self.has_pe:
            x = self.pe(x) # add position encoding to word embeddings
            if self.qpe:
                x = self.pe_listener(x)

        # if self.layers._modules["0"].norm1.stage == QuantStage.Quantized:
        #     assert x.quantized
        #     assert self.quantStub.stage == QuantStage.Quantized
        #     assert self.layers._modules["0"].mixer.stage == QuantStage.Quantized

        if self.quantStub.stage == QuantStage.Quantized:
            assert x.quantized
            assert not isinstance(mask, Tensor)
        elif self.quantStub.stage == QuantStage.QAT:
            pass

        for layer in self.layers:
            x = layer(x, mask)

        return x

    def __str__(self):
        if not self.layers[0].fft:
            s = f"{self.__class__.__name__}:(\n"
            s += f"\t(quantStub): {self.quantStub}\n"
            s += f"\t(embedding): {self.embedding}\n"
            if self.has_pe:
                s += f"\t(pe): {self.pe}\n"
            s += f"\t(emb_dropout): {self.emb_dropout}\n"
            s += str(self.layers)
            s += ")"
        else:
            s = "%s(num_layers=%r, fft=True)" % (
                self.__class__.__name__, len(self.layers)
            )
        return s

class QTSTModel(nn.Module):
    """
    Quantizable Classification/regression with Time Series Transformer.
    Structure from https://github.com/dhlee347/pytorchic-bert/blob/master/classify.py
    """
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
            learnable_label: bool = False,
            **kwargs # settings for quantizable modules
        ):
        super().__init__()

        self.learnable_label = learnable_label

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
                time_window=time_window,
                activ=activ,
                fft=fft,
                qkwargs=qkwargs,
                learnable_label=learnable_label,
            )

        self.task = task = task.lower()

        head = [nn.Dropout(fc_dropout)] # from https://github.com/timeseriesAI/tsai/blob/main/tsai/models/TST.py

        if "pre" in task:
            # pretrain: reconstruct input
            head += [
                QLinear(dim, src_dim, qkwargs=qkwargs),
            ]
            head += [
                QListener(head[-1], plot_name="head", **qkwargs),
                DeQuantStub(**qkwargs),
            ]

        elif "reg" in task:
            # regression: predict n scalars
            head += [
                nn.Flatten(1),
                QLinear(time_window * dim, n_labels, qkwargs=qkwargs),
            ]
            head += [
                QListener(head[-1], plot_name="head", **qkwargs),
                DeQuantStub(**qkwargs),
            ]
        elif "cl" in task:
            # classification: predict distribution over n labels (Softmax in CrossEntropyLoss)
            if not self.learnable_label:
                head += [
                    nn.Flatten(1),
                    QLinear(time_window * dim, n_labels, qkwargs=qkwargs),
                ]
            else:
                head += [
                    QLinear(dim, n_labels, qkwargs=qkwargs),
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
        self.freeze = freeze

    def forward(self, src, mask=None):

        if self.freeze:
            with torch.no_grad():
                h = self.encoder(src, mask)
        else:
            h = self.encoder(src, mask)

        # use the concatenated label as input to the head?
        head_inp = h[:,-1,:] if self.learnable_label else h
        out = self.head(head_inp)

        return out

SYMMETRIZING_MODULES += [
    QPositionalEncoding,
]


