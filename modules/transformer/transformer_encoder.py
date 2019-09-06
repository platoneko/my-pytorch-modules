import torch
import torch.nn as nn
import math
import numpy as np

from modules.transformer.multi_head_attention import MultiHeadAttention
from modules.transformer.transformer_ffn import TransformerFFN
from modules.utils import *


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
            dropout=0.0,
            attention_dropout=None,
            relu_dropout=None,
            learn_position_embedding=False,
            embedding_scale=False,
            reduction=False,
            num_positions=1024
    ):
        """

        :param num_heads: ``int``, required.
            Number of multihead attention heads.
        :param num_layers: ``int``, required.
            Number of transformer layers.
        :param embedding_size: ``int``, required.
            Must be a multiple of n_heads.
        :param embedding: ``torch.nn.Embedding``, required.
            An embedding matrix for the bottom layer of the transformer.
        :param ffn_size: ``int``,  required.
            The size of the hidden layer in the FFN.
        :param dropout: ``float``, optional (default = 0.0)
            Dropout used around embedding and before layer normalizations.
            This is used in Vaswani 2017 and works well on large datasets.
        :param attention_dropout: ``float``, optional (default = `dropout`)
            Dropout performed after the multi-head attention softmax.
        :param relu_dropout: ``float``, optional (default = `dropout`)
            Dropout used after the ReLU in the FFN.
            Not usedin Vaswani 2017, but used in Tensor2Tensor.
        :param learn_position_embedding: ``bool``, optional (default = False)
            If off, sinusoidal embedding are used.
            If on, position embedding are learned from scratch.
        :param embedding_scale: ``bool``, optional (default = False)
            Scale embedding relative to their dimensionality. Found useful in fairseq.
        :param reduction: ``bool``, optional (default = False)
            If true, returns the mean vector for the entire encoding sequence.
        :param num_positions: ``int``, optional (default = 1024)
            Max position of the position embedding matrix.
        """

        super().__init__()

        self.embedding_size = embedding_size
        self.embedding = embedding
        self.ffn_size = ffn_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim = embedding_size
        self.embedding_scale = embedding_scale
        self.reduction = reduction
        self.norm = nn.LayerNorm(embedding_size)
        if relu_dropout is None:
            relu_dropout = dropout
        if attention_dropout is None:
            attention_dropout = dropout

        self.out_dim = embedding_size
        assert embedding_size % num_heads == 0, \
            'Transformer embedding size must be a multiple of n_heads'

        # create the positional embedding
        self.position_embedding = nn.Embedding(num_positions, embedding_size)
        if not learn_position_embedding:
            create_position_embedding(num_positions, embedding_size, self.position_embedding.weight)
        else:
            nn.init.normal_(self.position_embedding.weight, 0, math.sqrt(embedding_size))

        # build the model
        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.layers.append(
                TransformerEncoderLayer(
                    num_heads,
                    embedding_size,
                    ffn_size,
                    attention_dropout=attention_dropout,
                    relu_dropout=relu_dropout,
                    dropout=dropout)
            )

    def forward(self, input, mask):
        """

        :param input: ``torch.LongTensor``, required.
            Input tokens tensor of shape (batch_size, seq_len).
        :param mask: ``torch.LongTensor``, required.
            Mask tensor of shape (batch_size, seq_len).
        :return:
            output: ``torch.FloatTensor``.
            A ``torch.FloatTensor`` of shape (batch_size, seq_len, embedding_size).
            If `reduction` is `True`, output is of shape (batch_size, embedding_size).
        """

        positions = (mask.cumsum(dim=1, dtype=torch.long) - 1).clamp_(min=0)
        tensor = self.embedding(input)
        if self.embedding_scale:
            tensor = tensor / np.sqrt(self.dim)
        tensor = tensor + self.position_embedding(positions).expand_as(tensor)
        for layer in self.layers:
            tensor = layer(tensor, mask)
        tensor = self.norm(tensor)
        if self.reduction:
            output = masked_mean(tensor, mask.unsqueeze(2), dim=1)
        else:
            output = tensor
        return output


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
        self.norm = nn.LayerNorm(embedding_size)
        self.ffn = TransformerFFN(embedding_size, ffn_size, dropout=relu_dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input, mask):
        # `input` of shape (batch_size, seq_len, embedding_size)
        input_norm = self.norm(input)
        context = self.attention(input_norm, input_norm, input_norm, mask=mask)
        output = self.ffn(self.dropout(context) + input)
        return output
