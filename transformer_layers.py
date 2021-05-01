# -*- coding: utf-8 -*-

from utils import noNaNs

from typing import Callable, Optional

from functools import partial

import math

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .quantizable_layer import \
        QListener, \
        QAdd, QMatMul, QScale, \
        QSoftmax


# this file contains quantized versions of layers used in tst_pytorch/modules.py


class QPositionalEncoding(nn.Module):
    """
    Learnable Position Encoding (A W x D bias matrix that we add to the input)
    """
    def __init__(self,
                 dim: int = 0,
                 time_window: int = 24,
                 init_fn: Callable = torch.rand
                 ):
        super().__init__()

        self.W = nn.Parameter(init_fn(time_window,dim))

    def forward(self, X):
        """
        Encode inputs.
        Args:
            X (FloatTensor): Sequence of word vectors
                ``(batch_size, dim, time_window)``
        """
        # Add position encodings
        out = self.W + X
        return out


class QMultiHeadedAttention(nn.Module):
    """
    Q Multi-Head Attention module from "Attention is All You Need"

    Implementation modified from OpenNMT-py.
    https://github.com/OpenNMT/OpenNMT-py

    (from https://github.com/joeynmt/joeynmt/blob/master/joeynmt/transformer_layers.py)
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

        self.pre_kq = QListener()
        self.k_layer = nn.Linear(dim, num_heads * head_size)
        self.post_kq = QListener(self.k_layer)

        self.pre_vq = QListener()
        self.v_layer = nn.Linear(dim, num_heads * head_size)
        self.post_vq = QListener(self.v_layer)

        self.pre_qq = QListener()
        self.q_layer = nn.Linear(dim, num_heads * head_size)
        self.post_qq = QListener(self.q_layer)

        self.scale = QScale()

        self.qkMatMul = QMatMul()

        self.qMask = QMask()

        self.qkMatMul = QMatMul()

        self.qsoftmax = QSoftmax(dim=-1)

        self.output_layer = nn.Linear(dim, dim)

    def forward(self, k: Tensor, v: Tensor, q: Tensor, mask: Tensor = None):
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
        k = self.pre_kq(k)
        k = self.k_layer(k)
        k = self.post_kq(k)

        v = self.pre_vq(v)
        v = self.v_layer(v)
        v = self.post_vq(v)

        q = self.pre_qq(q)
        q = self.q_layer(q)
        q = self.post_qq(q)

        # reshape q, k, v for our computation to (batch_size, num_heads, ..., ...)
        k = k.view(batch_size, num_heads, self.head_size, -1)
        v = v.view(batch_size, num_heads, -1, self.head_size)
        q = q.view(batch_size, num_heads, -1, self.head_size)

        # compute scores
        q = self.scale(q, math.sqrt(self.head_size))

        # batch x num_heads x query_len x key_len
        scores = self.qkMatMul(q, k)

        # apply the mask (if we have one)
        # we add a dimension for the heads to it below: [B, 1, 1, W]
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
            scores = self.qMask(scores, mask)
            # scores = scores.masked_fill(mask==torch.as_tensor(False), float('-inf'))

        # normalize context vectors.
        attention = self.qsoftmax(scores)

        # get context vector (select values with attention) and reshape
        # back to [B, W, D]
        context = self.vMatMul(attention, v)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, num_heads * self.head_size)

        output = self.output_layer(context)

        return output


class QPositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-forward layer
    Projects to ff_size and then back down to input_size.
    """

    def __init__(self, input_size, ff_size, time_window, dropout=0.1, activ=GeLU):
        """
        Initializes position-wise feed-forward layer.
        :param input_size: dimensionality of the input.
        :param ff_size: dimensionality of intermediate representation
        :param dropout:
        """
        super().__init__()
        modules = [
            nn.Linear(input_size, ff_size),
            activ(),
        ]
        modules += [
            QListener(modules[-1]),
            nn.Linear(ff_size, input_size) # FIXME needs next activ/quant
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
                 activ: nn.Module = GeLU,
                 **qkwargs
                 ):
        """
        A single Transformer layer.
        :param dim:
        :param ff_size:
        :param num_heads:
        :param dropout:
        :param qkwargs: keyword args QListener
        """
        super().__init__()



        self.src_src_att = MultiHeadedAttention(num_heads, dim,
                                                dropout=dropout, **qkwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.add1 = QAdd()
        self.qRes = QListener(self.add1, **qkwargs)
        self.batch_norm1 = BatchNorm1dTranspose(dim, momentum=bn_mom)
        self.feed_forward = PositionwiseFeedForward(dim, ff_size=ff_size,
                                                    dropout=dropout,
                                                    time_window=time_window,
                                                    activ=activ)
        self.dropout2 = nn.Dropout(dropout)
        self.add2 = QAdd()
        self.identity2 = QListener(self.add2)
        self.batch_norm2 = BatchNorm1dTranspose(dim, momentum=bn_mom)

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
        h = self.src_src_att(x, x, x, mask)

        sum1 = self.add1(h, self.dropout1(x))
        bn1inp = self.identity1(sum1)
        h = self.batch_norm1(bn1inp)

        ff_out = self.feed_forward(h)

        sum2 = self.add2(ff_out, self.dropout2(x))
        bn2inp = self.identity2(sum2)
        o = self.batch_norm2(bn2inp)

        return o


class QTransformerEncoder(nn.Module):
    """
    Transformer Encoder.
    From https://github.com/joeynmt/joeynmt/blob/master/joeynmt/encoders.py
    """

    def __init__(self,
                 src_dim: int = 187,
                 dim: int = 512,
                 ff_size: int = 2048,
                 num_layers: int = 8,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 emb_dropout: float = 0.1,
                 freeze: bool = False,
                 time_window: int = 24,
                 activ: nn.Module = GeLU,
                 **qkwargs):
        """
        Initializes the Transformer. NOTE: you still have to set self.mu, self.sigma later!
        :param src_dim: dimensionality of data
        :param dim: hidden size and size of embeddings
        :param ff_size: position-wise feed-forward layer size.
          (Typically this is 2*dim.)
        :param num_layers: number of layers
        :param num_heads: number of heads for multi-headed attention
        :param dropout: dropout probability for Transformer layers
        :param emb_dropout: Is applied to the input (word embeddings).
        :param freeze: freeze the parameters of the encoder during training
        :param qkwargs:
        """
        super().__init__()

        self.embedding = nn.Linear(src_dim, dim)
        self.pe = PositionalEncoding(dim, time_window)
        self.emb_dropout = nn.Dropout(p=emb_dropout)

        # build all (num_layers) layers
        self.layers = nn.ModuleList([
            QTransformerEncoderLayer(dim=dim, ff_size=ff_size,
                num_heads=num_heads, dropout=dropout,
                time_window=time_window, activ=activ, **qkwargs)
            for _ in range(num_layers)])

        self.batch_norm = BatchNorm1dTranspose(dim)

        if freeze:
            raise NotImplementedError("TODO Implement Freezing as in Paper")
            # freeze_params(self)

    def forward(self,
                src: Tensor,
                mask: Optional[Tensor] = None,
                **kwargs) -> (Tensor, Tensor):
        """
        Pass the input (and mask) through each layer in turn.
        Applies a Transformer encoder to sequence of embeddings x.
        The input mini-batch x needs to be sorted by src length.
        x and mask should have the same dimensions [batch, time, dim].

        :param src: embedded src inputs,
            shape (batch_size, src_len, embed_size)
        :param mask: indicates padding areas (zeros where padding), shape
            (batch_size, 1, src_len)
        :return:
            - output: hidden states with
                shape (batch_size, max_length, directions*hidden),
            - hidden_concat: last hidden state with
                shape (batch_size, directions*hidden)
        """

        src_embedded = self.embedding(src)
        x = self.pe(src_embedded)  # add position encoding to word embeddings
        x = self.emb_dropout(x)

        for layer in self.layers:
            x = layer(x, mask)

        return x

    def __repr__(self):
        return "%s(num_layers=%r, num_heads=%r)" % (
            self.__class__.__name__, len(self.layers),
            self.layers[0].src_src_att.num_heads)

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
            activ: nn.Module = GeLU,
            encoder: TransformerEncoder = None,
            time_window: int = 24,
            task: str = "pretrain", # regression/pretraining/classification
            n_labels: int = 10, # classification only
            fc_dropout: float = 0.0,
            **qkwargs
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
                **qkwargs
            )

        self.task = task = task.lower()

        head_list = [nn.Dropout(fc_dropout)] if fc_dropout else [] # from https://github.com/timeseriesAI/tsai/blob/main/tsai/models/TST.py

        if "pre" in task:
            # pretrain: reconstruct input
            head_list += [nn.Linear(dim, src_dim)]

        elif "reg" in task:
            # regression: predict n scalars
            head_list += [
                nn.Flatten(1),
                nn.Linear(time_window * dim, n_labels),
                nn.LogSoftmax(dim=-1),
            ]
        elif "cl" in task:
            # classification: predict distribution over n labels (Softmax in CrossEntropyLoss)
            head_list = [
                nn.Flatten(1),
                nn.Linear(time_window * dim, n_labels),
                nn.LogSoftmax(dim=-1),
            ]
        else:
            raise NotImplementedError(task)

        self.head = nn.Sequential(
            *head_list
        )


    def forward(self, src, mask=None):
        h = self.encoder(src, mask)
        out = self.head(h)
        return out



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
