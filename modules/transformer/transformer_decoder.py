import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from modules.utils import create_position_embedding, sequence_norm
from modules.transformer import MultiHeadAttention
from modules.transformer.transformer_ffn import TransformerFFN
from modules.beam_search import BeamSearch


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

    def forward(self, encoder_output, encoder_mask, target=None, num_steps=50, is_training=True):
        """
        forward

        :param
        encoder_output : ``torch.FloatTensor``, required.
            A ``torch.FloatTensor`` of shape (batch_size, enc_len, embedding_size)
        encoder_mask : ``torch.LongTensor``, required.
            A ``torch.LongTensor`` of shape (batch_size, enc_len)
        target : ``torch.LongTensor``, optional (default = None)
            Target tokens tensor of shape (batch_size, seq_len), `target` must contain <sos> and <eos> token.
        is_training : ``bool``, optional (default = True)
            If `True`, decode with a fixed, true target sequence.
        :return:
        tensor : ``torch.LongTensor``
            Return a tensor of shape (batch_size, seq_len),
        """
        if is_training:
            assert target is not None
            seq_len = target.size(1)
            positions = torch.arange(seq_len-1).unsqueeze(0)
            tensor = self.embedding(target[:, :-1].contiguous())
            if self.embedding_scale:
                tensor = tensor / np.sqrt(self.dim)
            tensor = tensor + self.position_embedding(positions).expand_as(tensor)
            tensor = self.dropout(tensor)  # --dropout
            for layer in self.layers:
                tensor = layer(tensor, encoder_output, encoder_mask, is_training)
            logits = self.output_layer(tensor)
        else:
            last_prediction = encoder_output.new_full((encoder_output.size(0),), fill_value=self.start_index).long()
            previous_input = None
            step_logits = []
            for timestep in range(num_steps):
                if (last_prediction == self.end_index).all():
                    break
                input = last_prediction
                output, previous_input = self._take_step(input, encoder_output, encoder_mask, previous_input)
                last_prediction = torch.argmax(output, dim=-1)
                step_logits.append(output)
            logits = torch.cat(step_logits, dim=1)

        return logits

    def _take_step(self, input, encoder_output, encoder_mask, previous_input=None):
        tensor = self.embedding(input).unsqueeze(1)
        if self.embedding_scale:
            tensor = tensor / np.sqrt(self.dim)
        if previous_input is not None:
            tensor = tensor + self.position_embedding(previous_input.size(1)).expand_as(tensor)
            decoder_input = torch.cat([previous_input, tensor], dim=1)
        else:
            tensor = tensor + self.position_embedding(0).expand_as(tensor)
            decoder_input = tensor
        for layer in self.layers:
            output = layer(decoder_input, encoder_output, encoder_mask, is_training=False)
        output = self.output_layer(output)
        return output, decoder_input

    def forward_beam_search(
            self, encoder_output, encoder_mask,
            num_steps=50, beam_size=4,per_node_beam_size=4,
    ):
        """
        Decoder forward using beam search at inference stage

        :param
        beam_size : ``int``, optional (default = 4)
        per_node_beam_size : ``int``, optional (default = 4)

        :return
        all_top_k_predictions : ``torch.LongTensor``
            A ``torch.LongTensor`` of shape (batch_size, beam_size, num_steps),
            containing k top sequences in descending order along dim 1.
        log_probabilities : ``torch.FloatTensor``
            A ``torch.FloatTensor``  of shape (batch_size, beam_size),
            Log probabilities of k top sequences.
        """

        beam_search = BeamSearch(self.end_index, num_steps, beam_size, per_node_beam_size)
        start_prediction = encoder_output.new_full((encoder_output.size(0),), fill_value=self.start_index).long()

        state = {'encoder_output': encoder_output, 'encoder_mask': encoder_mask}
        all_top_k_predictions, log_probabilities = \
            beam_search.search(start_prediction, state, self._beam_step)
        return all_top_k_predictions, log_probabilities

    def _beam_step(self, input, state):
        encoder_output = state['encoder_output']
        encoder_mask = state['encoder_mask']
        tensor = self.embedding(input).unsqueeze(1)
        if self.embedding_scale:
            tensor = tensor / np.sqrt(self.dim)
        if 'previous_input' in state:
            previous_input = state['previous_input']
            tensor = tensor + self.position_embedding(previous_input.size(1)).expand_as(tensor)
            decoder_input = torch.cat([previous_input, tensor.expand(previous_input.size(0), -1, -1)], dim=1)
        else:
            tensor = tensor + self.position_embedding(0).expand_as(tensor)
            decoder_input = tensor
        state['previous_input'] = decoder_input
        for layer in self.layers:
            output = layer(decoder_input, encoder_output, encoder_mask, is_training=False)
        log_prob = F.log_softmax(self.output_layer(output), dim=-1)
        return log_prob, state


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
