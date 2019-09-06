import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from modules.transformer.multi_head_attention import MultiHeadAttention
from modules.transformer.transformer_ffn import TransformerFFN
from modules.beam_search import BeamSearch
from modules.utils import *


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
            start_index,
            end_index,
            output_layer,
            dropout=0.0,
            attention_dropout=None,
            relu_dropout=None,
            embedding_scale=True,
            learn_positional_embedding=False,
            num_positions=1024,
    ):
        """

        :param num_heads: ``int``, required.
            Number of multihead attention heads.
        :param num_layers: ``int``, required.
            Number of transformer layers.
        :param embedding_size: ``int``, required.
            Must be a multiple of n_heads.
        :param ffn_size: ``int``,  required.
            The size of the hidden layer in the FFN.
        :param embedding: ``torch.nn.Embedding``, required.
            An embedding matrix for the bottom layer of the transformer.
        :param start_index: ``int``, required.
            Index of start token.
        :param end_index: ``int``, required.
            Index of end token.
        :param output_layer: ``torch.Modules``, required.
            Output layer to projecting transformer output to logits.
        :param dropout: ``float``, optional (default = 0.0)
            Dropout used around embedding and before layer normalizations.
            This is used in Vaswani 2017 and works well on large datasets.
        :param attention_dropout: ``float``, optional (default = `dropout`)
            Dropout performed after the multi-head attention softmax.
        :param relu_dropout: ``float``, optional (default = `dropout`)
            Dropout used after the ReLU in the FFN.
            Not usedin Vaswani 2017, but used in Tensor2Tensor.
        :param embedding_scale: ``bool``, optional (default = False)
            Scale embedding relative to their dimensionality. Found useful in fairseq.
        :param learn_positional_embedding: ``bool``, optional (default = False)
            If off, sinusoidal embedding are used.
            If on, position embedding are learned from scratch.
        :param num_positions: ``int``, optional (default = 1024)
            Max position of the position embedding matrix.
        """

        super().__init__()
        self.embedding_size = embedding_size
        self.ffn_size = ffn_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim = embedding_size
        self.embedding_scale = embedding_scale
        self.start_index = start_index
        self.end_index = end_index
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
        for _ in range(self.num_layers):
            self.layers.append(
                TransformerDecoderLayer(
                    num_heads, embedding_size, ffn_size,
                    attention_dropout=attention_dropout,
                    relu_dropout=relu_dropout,
                    dropout=dropout,
                )
            )
        self.output_layer = output_layer
        self.norm = nn.LayerNorm(embedding_size)

    def forward(self, encoder_output, encoder_mask, target=None, num_steps=50, is_training=True):
        """

        :param encoder_output: ``torch.FloatTensor``, required.
            A ``torch.FloatTensor`` of shape (batch_size, enc_len, embedding_size)
        :param encoder_mask: ``torch.LongTensor``, required.
            A ``torch.LongTensor`` of shape (batch_size, enc_len)
        :param target: ``torch.LongTensor``, optional (default = None)
            Target tokens tensor of shape (batch_size, seq_len), ``target`` must contain start and end token.
        :param num_steps: ``int``, optional (default = 50)
            Number of decoding steps.
        :param is_training: ``bool``, optional (default = True)
            During training stage(training and validation), we always feed ground truth token to generate next token.
        :return:
            logits: ``torch.FloatTensor``
            An unnormalized ``torch.FloatTensor`` of shape (batch_size, tgt_len/num_steps, vocab_size)
        """

        if is_training:
            assert target is not None
            seq_len = target.size(1)
            positions = get_range_vector(seq_len-1, get_device_of(encoder_output))
            tensor = self.embedding(target[:, :-1].contiguous())
            if self.embedding_scale:
                tensor = tensor / np.sqrt(self.dim)
            tensor = tensor + self.position_embedding(positions).expand_as(tensor)
            for layer in self.layers:
                tensor = layer(tensor, encoder_output, encoder_mask, is_training)
            tensor = self.norm(tensor)
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
            position = input.new_full((input.size(0), 1), fill_value=previous_input.size(1))
            tensor = tensor + self.position_embedding(position)
            decoder_input = torch.cat([previous_input, tensor], dim=1)
        else:
            position = input.new_full((input.size(0), 1), fill_value=0)
            tensor = tensor + self.position_embedding(position)
            decoder_input = tensor
        for layer in self.layers:
            output = layer(decoder_input, encoder_output, encoder_mask, is_training=False)
        output = self.norm(output)
        output = self.output_layer(output)
        return output, decoder_input

    def beam_forward(
            self, encoder_output, encoder_mask,
            num_steps=50, beam_size=4, per_node_beam_size=None,
    ):
        """

        :param encoder_output: ``torch.FloatTensor``, required.
            A ``torch.FloatTensor`` of shape (batch_size, enc_len, embedding_size)
        :param encoder_mask: ``torch.LongTensor``, required.
            A ``torch.LongTensor`` of shape (batch_size, enc_len)
        :param num_steps: ``int``, optional (default = 50)
            Number of decoding steps.
        :param beam_size: ``int``, optional (default = 4)
            The width of the beam used.
        :param per_node_beam_size: ``int``, optional (default = ``beam_size``)
            The maximum number of candidates to consider per node, at each step in the search.
            If not given, this just defaults to ``beam_size``. Setting this parameter
            to a number smaller than ``beam_size`` may give better results, as it can introduce
            more diversity into the search. See `Beam Search Strategies for Neural Machine Translation.
            Freitag and Al-Onaizan, 2017 <http://arxiv.org/abs/1702.01806>`.
        :return:
            all_top_k_predictions: ``torch.LongTensor``
            A ``torch.LongTensor`` of shape (batch_size, beam_size, num_steps),
            containing k top sequences in descending order along dim 1.
            log_probabilities: ``torch.FloatTensor``
            A ``torch.FloatTensor``  of shape (batch_size, beam_size),
            Log probabilities of k top sequences.
        """

        if per_node_beam_size is None:
            per_node_beam_size = beam_size
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
            position = input.new_full((input.size(0), 1), fill_value=previous_input.size(1))
            tensor = tensor + self.position_embedding(position)
            decoder_input = torch.cat([previous_input, tensor.expand(previous_input.size(0), -1, -1)], dim=1)
        else:
            position = input.new_full((input.size(0), 1), fill_value=0)
            tensor = tensor + self.position_embedding(position)
            decoder_input = tensor
        state['previous_input'] = decoder_input
        for layer in self.layers:
            output = layer(decoder_input, encoder_output, encoder_mask, is_training=False)
        output = self.norm(output)
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

    def forward(self, input, encoder_output, encoder_mask, is_training):
        """

        :param input: ``torch.FloatTensor``, required.
            A ``torch.FloatTensor`` of shape (batch_size, seq_len, embedding_size).
        :param encoder_output: ``torch.FloatTensor``, required.
            A ``torch.FloatTensor`` of shape (batch_size, enc_len, embedding_size).
        :param encoder_mask: ``torch.LongTensor``, required.
            A ``torch.LongTensor``of shape (batch_size, enc_len).
        :param is_training: ``bool``, required.
            During training stage(training and validation), as we know ground truth target,
            we can directly compute the whole predict sequence.
            However, during test stage, we must feed the last prediction token to generate next token,
            so this module can only compute one next token.
        :return:
            output: ``torch.FloatTensor``
            If ``is_training`` is True, the output is of shape (batch_size, tgt_len, embedding_size),
            Otherwise, the output is of shape (batch_size, 1, embedding_size)
        """

        input_norm = self.norm1(input)
        if is_training:
            decoder_mask = self._create_selfattn_mask(input)
            # first self attn
            # don't peak into the future!
            query = self.self_attention(input_norm, input_norm, input_norm, mask=decoder_mask)
        else:
            query = self.self_attention(input_norm[:, -1, :].unsqueeze(1), input_norm, input_norm)

        query = self.dropout(query) + input
        query_norm = self.norm2(query)
        inter = self.encoder_attention(
            query=query_norm,
            key=encoder_output,
            value=encoder_output,
            mask=encoder_mask
        )
        output = self.ffn(self.dropout(inter) + query)

        return output

    def _create_selfattn_mask(self, x):
        # figure out how many timesteps we need
        bsz = x.size(0)
        time = x.size(1)
        # make sure that we don't look into the future
        mask = torch.tril(x.new(time, time).fill_(1))
        # broadcast across batch
        mask = mask.unsqueeze(0).expand(bsz, -1, -1)
        return mask
