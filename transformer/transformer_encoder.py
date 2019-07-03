import torch
import torch.nn as nn
import math
import numpy as np

from transformer import MultiHeadAttention
from transformer import TransformerFFN
from utils import create_positional_features, get_device_of


def _normalize(tensor, norm_layer):
    """
    Broadcast layer norm
    """
    size = tensor.size()
    return norm_layer(tensor.view(-1, size[-1])).view(size)


class TransformerEncoder(nn.Module):
    """
    Transformer encoder module.
    """

    def __init__(
            self,
            num_heads,
            num_layers,
            embedding_size,
            embedding,
            ffn_size,
            pad_index,
            dropout=0.0,
            attention_dropout=None,
            relu_dropout=None,
            learn_positional_embeddings=False,
            embeddings_scale=False,
            reduction=False,
            num_positions=1024
    ):
        """
        :param
        num_heads : ``int``, required.
            Number of multihead attention heads.
        num_layers : ``int``, required.
            Number of transformer layers.
        embedding_size : ``int``, required.
            Must be a multiple of n_heads.
        embedding : ``torch.nn.Embedding``, required.
            An embedding matrix for the bottom layer of the transformer.
        ffn_size : ``int``,  required.
            The size of the hidden layer in the FFN.
        pad_index : ``int``, required.
            Reserved padding index in the embeddings matrix.
        dropout : ``float``, optional (default = 0.0)
            Dropout used around embeddings and before layer normalizations.
            This is used in Vaswani 2017 and works well on large datasets.
        attention_dropout : ``float``, optional (default = `dropout`)
            Dropout performed after the multi-head attention softmax.
        relu_dropout : ``float``, optional (default = `dropout`)
            Dropout used after the ReLU in the FFN.
            Not usedin Vaswani 2017, but used in Tensor2Tensor.
        learn_positional_embeddings : ``bool``, optional (default = False)
            If off, sinusoidal embeddings are used.
            If on, position embeddings are learned from scratch.
        embeddings_scale : ``bool``, optional (default = False)
            Scale embeddings relative to their dimensionality. Found useful in fairseq.
        reduction : ``bool``, optional (default = False)
            If true, returns the mean vector for the entire encoding sequence.
        num_positions : ``int``, optional (default = 1024)
            Max position of the position embeddings matrix.
        """
        super().__init__()

        self.embedding_size = embedding_size
        self.embedding = embedding
        self.ffn_size = ffn_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.pad_index = pad_index
        self.dim = embedding_size
        self.embeddings_scale = embeddings_scale
        self.reduction = reduction
        # this is --dropout, not --relu-dropout or --attention-dropout
        self.dropout = nn.Dropout(p=dropout)
        if relu_dropout is None:
            relu_dropout = dropout
        if attention_dropout is None:
            attention_dropout = dropout

        self.out_dim = embedding_size
        assert embedding_size % num_heads == 0, \
            'Transformer embedding size must be a multiple of n_heads'

        # create the positional embeddings
        self.position_embeddings = nn.Embedding(num_positions, embedding_size)
        if not learn_positional_embeddings:
            self.position_embeddings.weight = create_positional_features(
                num_positions,
                embedding_size,
                get_device_of(self.position_embeddings.weight))
            self.position_embeddings.weight.requires_grad = False
        else:
            nn.init.normal_(self.position_embeddings.weight, 0, math.sqrt(embedding_size))

        # build the model
        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.layers.append(TransformerEncoderLayer(
                num_heads,
                embedding_size,
                ffn_size,
                attention_dropout=attention_dropout,
                relu_dropout=relu_dropout,
                dropout=dropout))

    def forward(self, inputs):
        """
        forward

        :param
        inputs : ``torch.LongTensor``, required.
            Input tokens tensor of shape (batch_size, seq_len).

        :return
        outputs : ``torch.FloatTensor``
            A ``torch.FloatTensor`` of shape (batch_size, seq_len, embedding_size).
            If `reduction` is `True`, outputs is of shape (batch_size, embedding_size).

        """
        mask = (inputs != self.pad_index)
        positions = (mask.cumsum(dim=1, dtype=torch.long) - 1).clamp_(min=0)
        tensor = self.embeddings(inputs)
        if self.embeddings_scale:
            tensor = tensor * np.sqrt(self.dim)
        tensor = tensor + self.position_embeddings(positions).expand_as(tensor)
        # --dropout on the embeddings
        tensor = self.dropout(tensor)

        tensor *= mask.unsqueeze(-1).float()
        for i in range(self.num_layers):
            tensor = self.layers[i](tensor, mask)

        if self.reduction:
            divisor = mask.float().sum(dim=1).unsqueeze(-1).clamp(min=1e-20)
            outputs = tensor.sum(dim=1) / divisor
            return outputs
        else:
            outputs = tensor
            return outputs, mask


class TransformerEncoderLayer(nn.Module):
    def __init__(
            self,
            num_heads,
            embedding_size,
            ffn_size,
            attention_dropout=0.0,
            relu_dropout=0.0,
            dropout=0.0,
    ):
        super().__init__()

        self.dim = embedding_size
        self.ffn_dim = ffn_size
        self.attention = MultiHeadAttention(
            num_heads, embedding_size,
            dropout=attention_dropout,  # --attention-dropout
        )
        self.norm1 = nn.LayerNorm(embedding_size)
        self.ffn = TransformerFFN(embedding_size, ffn_size, dropout=relu_dropout)
        self.norm2 = nn.LayerNorm(embedding_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, tensor, mask):
        # shape (batch_size, seq_len, embedding_size)
        tensor = tensor + self.dropout(self.attention(tensor, tensor, tensor, mask=mask))
        tensor = _normalize(tensor, self.norm1)
        tensor = tensor + self.dropout(self.ffn(tensor))
        tensor = _normalize(tensor, self.norm2)
        # processing padding
        tensor *= mask.unsqueeze(-1).float()
        return tensor
