import torch
import torch.nn as nn
import math
import numpy as np

from modules.utils import create_position_embedding, sequence_norm
from modules.transformer import MultiHeadAttention
from modules.transformer.transformer_ffn import TransformerFFN


class TransformerDecoder(nn.Module):
    """
    Transformer decoder module.
    """

    def __init__(
            self,
            num_heads,
            num_layers,
            embedding_size,
            ffn_size,
            embedding,
            output_layer,
            dropout=0.0,
            attention_dropout=None,
            relu_dropout=None,
            embedding_scale=True,
            learn_positional_embedding=False,
            num_positions=1024,
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
        dropout : ``float``, optional (default = 0.0)
            Dropout used around embedding and before layer normalizations.
            This is used in Vaswani 2017 and works well on large datasets.
        attention_dropout : ``float``, optional (default = `dropout`)
            Dropout performed after the multi-head attention softmax.
        relu_dropout : ``float``, optional (default = `dropout`)
            Dropout used after the ReLU in the FFN.
            Not usedin Vaswani 2017, but used in Tensor2Tensor.
        learn_position_embedding : ``bool``, optional (default = False)
            If off, sinusoidal embedding are used.
            If on, position embedding are learned from scratch.
        embedding_scale : ``bool``, optional (default = False)
            Scale embedding relative to their dimensionality. Found useful in fairseq.
        num_positions : ``int``, optional (default = 1024)
            Max position of the position embedding matrix.
        """

        super().__init__()
        self.embedding_size = embedding_size
        self.ffn_size = ffn_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim = embedding_size
        self.embedding_scale = embedding_scale
        self.dropout = nn.Dropout(p=dropout)  # --dropout
        if relu_dropout is None:
            relu_dropout = dropout
        if attention_dropout is None:
            attention_dropout = dropout

        self.out_dim = embedding_size

        assert embedding_size % num_heads == 0, \
            'Transformer embedding size must be a multiple of num_heads'

        self.embedding = embedding

        # create the positional embedding
        self.position_embedding = nn.Embedding(num_positions, embedding_size)
        if not learn_positional_embedding:
            create_position_embedding(num_positions, embedding_size, self.position_embedding.weight)
        else:
            nn.init.normal_(self.position_embedding.weight, 0, math.sqrt(embedding_size))

        # build the model
        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(
                TransformerDecoderLayer(
                    num_heads, embedding_size, ffn_size,
                    attention_dropout=attention_dropout,
                    relu_dropout=relu_dropout,
                    dropout=dropout,
                )
            )
        self.output_layer = output_layer

    def forward(self, input, encoder_output, encoder_mask, is_training=True):
        """
        forward

        :param
        input : ``torch.FloatTensor``, required.
            A ``torch.FloatTensor`` of shape (batch_size, seq_len, embedding_size)
        encoder_output : ``torch.FloatTensor``, required.
            A ``torch.FloatTensor`` of shape (batch_size, enc_len, embedding_size)
        encoder_mask : ``torch.ByteTensor``, required.
            A ``torch.ByteTensor`` of shape (batch_size, enc_len)
        is_training : ``bool``, optional (default = True)
            If `True`, decode with a fixed, true target sequence.
        :return:
        tensor : ``torch.LongTensor``
            Return a tensor of shape (batch_size, seq_len),
        """
        seq_len = input.size(1)
        positions = torch.arange(seq_len).unsqueeze(0)
        tensor = self.embedding(input)
        if self.embedding_scale:
            tensor = tensor / np.sqrt(self.dim)
        tensor = tensor + self.position_embedding(positions).expand_as(tensor)
        tensor = self.dropout(tensor)  # --dropout

        for layer in self.layers:
            tensor = layer(tensor, encoder_output, encoder_mask, is_training)
        
        return tensor


class TransformerDecoderLayer(nn.Module):
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
        self.dropout = nn.Dropout(p=dropout)

        self.self_attention = MultiHeadAttention(
            num_heads, embedding_size, dropout=attention_dropout
        )

        self.norm1 = nn.LayerNorm(embedding_size)

        self.encoder_attention = MultiHeadAttention(
            num_heads, embedding_size, dropout=attention_dropout
        )

        self.norm2 = nn.LayerNorm(embedding_size)

        self.ffn = TransformerFFN(embedding_size, ffn_size, dropout=relu_dropout)
        self.norm3 = nn.LayerNorm(embedding_size)

    def forward(self, x, encoder_output, encoder_mask, is_training):
        """

        :param x: ``torch.FloatTensor``, required.
         A ``torch.FloatTensor`` of shape (batch_size, seq_len, embedding_size).
        :param encoder_output: ``torch.FloatTensor``, required.
         A ``torch.FloatTensor`` of shape (batch_size, enc_len, embedding_size).
        :param encoder_mask: ``torch.LongTensor``, required.
         A ``torch.LongTensor``of shape (batch_size, enc_len).
        :param is_training: ``bool``
        :return:
        """

        residual = x

        if is_training:
            decoder_mask = self._create_selfattn_mask(x)
            # first self attn
            # don't peak into the future!
            x = self.self_attention(x, x, x, mask=decoder_mask)
            x = self.dropout(x)  # --dropout
            x = x + residual
            x = sequence_norm(x, self.norm1)  # (batch_size, seq_len, embedding_size)
        else:
            x = self.self_attention(x[:, -1, :].unsqueeze(1), x, x)
            x = self.dropout(x)
            x = x + residual
            x = sequence_norm(x, self.norm1)  # (batch_size, 1, embedding_size)

        residual = x

        x = self.encoder_attention(
            query=x,
            key=encoder_output,
            value=encoder_output,
            mask=encoder_mask
        )
        x = self.dropout(x)  # --dropout
        x = residual + x
        x = sequence_norm(x, self.norm2)

        # finally the ffn
        residual = x
        x = self.ffn(x)
        x = self.dropout(x)  # --dropout
        x = residual + x
        x = sequence_norm(x, self.norm3)

        return x

    def _create_selfattn_mask(self, x):
        # figure out how many timesteps we need
        bsz = x.size(0)
        time = x.size(1)
        # make sure that we don't look into the future
        mask = torch.tril(x.new(time, time).fill_(1))
        # broadcast across batch
        mask = mask.unsqueeze(0).expand(bsz, -1, -1)
        return mask
