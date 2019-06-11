import torch
import torch.nn as nn
import math
import numpy as np

from utils import create_positional_features, get_device_of
from transformer import MultiHeadAttention
from transformer import TransformerFFN


def _normalize(tensor, norm_layer):
    """
    Broadcast layer norm
    """
    size = tensor.size()
    return norm_layer(tensor.view(-1, size[-1])).view(size)


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder layer.
     :param int n_heads: the number of multihead attention heads.
    :param int n_layers: number of transformer layers.
    :param int embedding_size: the embedding sizes. Must be a multiple of n_heads.
    :param int ffn_size: the size of the hidden layer in the FFN
    :param embedding: an embedding matrix for the bottom layer of the transformer.
        If none, one is created for this encoder.
    :param float dropout: Dropout used around embeddings and before layer
        layer normalizations. This is used in Vaswani 2017 and works well on
        large datasets.
    :param float attention_dropout: Dropout performed after the multhead attention
        softmax. This is not used in Vaswani 2017.
    :param float relu_attention: Dropout used after the ReLU in the FFN. Not used
        in Vaswani 2017, but used in Tensor2Tensor.
    :param int padding_idx: Reserved padding index in the embeddings matrix.
    :param bool learn_positional_embeddings: If off, sinusoidal embeddings are
        used. If on, position embeddings are learned from scratch.
    :param bool embeddings_scale: Scale embeddings relative to their dimensionality.
        Found useful in fairseq.
    :param int n_positions: Size of the position embeddings matrix.
    """

    def __init__(
        self,
        num_heads,
        num_layers,
        embedding_size,
        ffn_size,
        embedding,
        dropout=0.0,
        attention_dropout=None,
        relu_dropout=None,
        embeddings_scale=True,
        learn_positional_embeddings=False,
        num_positions=1024,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.ffn_size = ffn_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim = embedding_size
        self.embeddings_scale = embeddings_scale
        self.dropout = nn.Dropout(p=dropout)  # --dropout
        if relu_dropout is None:
            relu_dropout = dropout
        if attention_dropout is None:
            attention_dropout = dropout

        self.out_dim = embedding_size

        assert embedding_size % num_heads == 0, \
            'Transformer embedding size must be a multiple of num_heads'

        self.embedding = embedding

        # create the positional embeddings
        self.position_embeddings = nn.Embedding(num_positions, embedding_size)
        if not learn_positional_embeddings:
            self.position_embeddings.weight = create_positional_features(
                num_positions,
                embedding_size,
                get_device_of(self.position_embeddings.weight))
        else:
            nn.init.normal_(self.position_embeddings.weight, 0, math.sqrt(embedding_size))

        # build the model
        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(TransformerDecoderLayer(
                num_heads, embedding_size, ffn_size,
                attention_dropout=attention_dropout,
                relu_dropout=relu_dropout,
                dropout=dropout,
            ))

    def forward(self, inputs, encoder_outputs, encoder_mask, force=True):
        """
        forward

        :param
        inputs : ``torch.FloatTensor``, required.
            Target sequence in train, while generated sequence at current time step in dev/test.
            A ``torch.FloatTensor`` of shape (batch_size, seq_len, embedding_size)
        encoder_outputs : ``torch.FloatTensor``, required.
            A ``torch.FloatTensor`` of shape (batch_size, enc_len, embedding_size)
        encoder_mask : ``torch.ByteTensor``, required.
            A ``torch.ByteTensor`` of shape (batch_size, enc_len)
        force : ``bool``, optional (default = True)
            If `True`, decode with a fixed, true target sequence.
        :return:
        tensor : ``torch.FloatTensor``
            If `force` is `True`, return a tensor of shape (batch_size, seq_len, embedding_size),
            else return a tensor of shape (batch_size, embedding_size)
        """
        seq_len = inputs.size(1)
        positions = torch.arange(seq_len).unsqueeze(0)
        tensor = self.embeddings(inputs)
        if self.embeddings_scale:
            tensor = tensor * np.sqrt(self.dim)
        tensor = tensor + self.position_embeddings(positions).expand_as(tensor)
        tensor = self.dropout(tensor)  # --dropout

        for layer in self.layers:
            tensor = layer(tensor, encoder_outputs, encoder_mask, force)

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

    def forward(self, x, encoder_output, encoder_mask, force):
        residual = x

        if force and self.training:
            decoder_mask = self._create_selfattn_mask(x)
            # first self attn
            # don't peak into the future!
            x = self.self_attention(x, x, x, mask=decoder_mask)
            x = self.dropout(x)  # --dropout
            x = x + residual
            x = _normalize(x, self.norm1)  # (batch_size, seq_len, embedding_size)
        else:
            x = self.self_attention(x[:, -1, :].unsqueeze(1), x, x)
            x = self.dropout(x)
            x = x + residual
            x = _normalize(x, self.norm1)  # (batch_size, 1, embedding_size)

        residual = x

        x = self.encoder_attention(
            query=x,
            key=encoder_output,
            value=encoder_output,
            mask=encoder_mask
        )
        x = self.dropout(x)  # --dropout
        x = residual + x
        x = _normalize(x, self.norm2)

        # finally the ffn
        residual = x
        x = self.ffn(x)
        x = self.dropout(x)  # --dropout
        x = residual + x
        x = _normalize(x, self.norm3)

        if not force or not self.training:
            x = x.squeeze(1)

        return x

    def _create_selfattn_mask(self, x):
        # figure out how many timestamps we need
        bsz = x.size(0)
        time = x.size(1)
        # make sure that we don't look into the future
        mask = torch.tril(x.new(time, time).fill_(1))
        # broadcast across batch
        mask = mask.unsqueeze(0).expand(bsz, -1, -1)
        return mask
