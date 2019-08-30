import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.beam_search import BeamSearch
from modules.rnn.rnn_decoder import GRUDecoder, LSTMDecoder


class StackGRUDecoder(GRUDecoder):
    """
    A multi-layer GRU recurrent neural network decoder.
    """
    def __init__(
            self,
            input_size,
            hidden_size,
            start_index,
            end_index,
            embedding,
            output_layer,
            num_layers=2,
            attention=None,
            dropout=0.0
    ):
        super().__init__(
            input_size,
            hidden_size,
            start_index,
            end_index,
            embedding,
            output_layer,
            attention=attention,
            dropout=dropout
        )

        self.num_layers = num_layers

        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

    def _take_step(self, input, hidden, encoder_output=None, encoder_mask=None):
        # `input` of shape: (batch_size,)
        # `hidden` of shape: (num_layers, batch_size, hidden_size)
        # shape: (batch_size, input_size)
        embedded_input = self.embedding(input)
        rnn_input = embedded_input
        if self.attention is not None:
            # shape: (batch_size, num_rows)
            attn_score = self.attention(hidden[-1], encoder_output, encoder_mask)
            attn_input = attn_score.unsqueeze(1).matmul(encoder_output).squeeze(1)
            # shape: (batch_size, input_size + attn_size)
            rnn_input = torch.cat([embedded_input, attn_input], dim=-1)
        _, next_hidden = self.rnn(rnn_input.unsqueeze(1), hidden)
        # shape: (batch_size, vocab_size)
        output = self.output_layer(next_hidden[-1])
        return output, next_hidden

    def beam_forward(
            self,
            hidden,
            encoder_output=None,
            encoder_mask=None,
            num_steps=50,
            beam_size=4,
            per_node_beam_size=4
    ):
        """
        Decoder forward using beam search at inference stage

        :param
        hidden : ``torch.FloatTensor``, required.
            Initial hidden tensor of shape (num_layers, batch_size, hidden_size)
        end_index : ``int``, required.
            Vocab index of <eos> token.
        encoder_output : ``torch.FloatTensor``, optional (default = None)
            A ``torch.FloatTensor`` of shape (batch_size, num_rows, value_size)
        encoder_mask : ``torch.LongTensor``, optional (default = None)
            A ``torch.LongTensor`` of shape (batch_size, num_rows)
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

        if self.attention is not None:
            assert encoder_output is not None

        beam_search = BeamSearch(self.end_index, num_steps, beam_size, per_node_beam_size)
        start_prediction = hidden.new_full((hidden.size(1),), fill_value=self.start_index).long()

        # `hidden` of shape: (batch_size, num_layers, hidden_size)
        hidden = hidden.transpose(0, 1).contiguous()
        state = {'hidden': hidden}
        if self.attention:
            state['encoder_output'] = encoder_output
            state['encoder_mask'] = encoder_mask
        all_top_k_predictions, log_probabilities = \
            beam_search.search(start_prediction, state, self._beam_step)
        return all_top_k_predictions, log_probabilities

    def _beam_step(self, input, state):
        # shape: (group_size, input_size)
        embedded_input = self.embedding(input)
        rnn_input = embedded_input

        # shape: (group_size, num_layers, input_size)
        hidden = state['hidden']
        # shape: (num_layers, group_size, input_size)
        hidden = hidden.transpose(0, 1).contiguous()
        if self.attention is not None:
            encoder_output = state['encoder_output']
            encoder_mask = state['encoder_mask']
            # shape: (group_size, num_rows)
            attn_score = self.attention(hidden[-1], encoder_output, encoder_mask)
            attn_input = attn_score.unsqueeze(1).matmul(encoder_output).squeeze(1)
            # shape: (group_size, input_size + attn_size)
            rnn_input = torch.cat([embedded_input, attn_input], dim=-1)
        _, next_hidden = self.rnn(rnn_input.unsqueeze(1), hidden)
        state['hidden'] = next_hidden.transpose(0, 1).contiguous()
        # shape: (group_size, vocab_size)
        log_prob = F.log_softmax(self.output_layer(next_hidden[-1]), dim=-1)
        return log_prob, state


class StackLSTMDecoder(LSTMDecoder):
    """
    A multi-layer LSTM recurrent neural network decoder.
    """
    def __init__(
            self,
            input_size,
            hidden_size,
            start_index,
            end_index,
            embedding,
            output_layer,
            num_layers=2,
            attention=None,
            dropout=0.0
    ):
        super().__init__(
            input_size,
            hidden_size,
            start_index,
            end_index,
            embedding,
            output_layer,
            attention=attention,
            dropout=dropout
        )

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.start_index = start_index
        self.end_index = end_index
        self.embedding = embedding
        self.attention = attention
        self.dropout = dropout
        self.num_layers = num_layers

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

    def _take_step(self, input, hidden_tuple, encoder_output=None, encoder_mask=None):
        # `input` of shape: (batch_size,)
        # `hidden` & `cell_state` of shape: (num_layers, batch_size, hidden_size)
        # shape: (batch_size, input_size)
        embedded_input = self.embedding(input)
        rnn_input = embedded_input
        if self.attention is not None:
            # shape: (batch_size, num_rows)
            attn_score = self.attention(hidden_tuple[0][-1], encoder_output, encoder_mask)
            attn_input = attn_score.unsqueeze(1).matmul(encoder_output).squeeze(1)
            # shape: (batch_size, input_size + attn_size)
            rnn_input = torch.cat([embedded_input, attn_input], dim=-1)
        _, (next_hidden, next_cell_state) = self.rnn(rnn_input.unsqueeze(1), hidden_tuple)
        # shape: (batch_size, vocab_size)
        output = self.output_layer(next_hidden[-1])
        return output, (next_hidden, next_cell_state)

    def beam_forward(
            self,
            hidden,
            encoder_output=None,
            encoder_mask=None,
            num_steps=50,
            beam_size=4,
            per_node_beam_size=4,
            early_stop=False):
        """
        Decoder forward using beam search at inference stage

        :param
        hidden : ``torch.FloatTensor``, required.
            Initial hidden tensor of shape (batch_size, hidden_size)
        end_index : ``int``, required.
            Vocab index of <eos> token.
        encoder_output : ``torch.FloatTensor``, optional (default = None)
            A ``torch.FloatTensor`` of shape (batch_size, num_rows, value_size)
        encoder_mask : ``torch.LongTensor``, optional (default = None)
            A ``torch.LongTensor`` of shape (batch_size, num_rows)
        beam_size : ``int``, optional (default = 4)
        per_node_beam_size : ``int``, optional (default = 4)
        early_stop : ``bool``, optional (default = False).
            If every predicted token from the last step is `self.end_index`, then we can stop early.

        :return
        all_top_k_predictions : ``torch.LongTensor``
            A ``torch.LongTensor`` of shape (batch_size, beam_size, num_steps),
            containing k top sequences in descending order along dim 1.
        log_probabilities : ``torch.FloatTensor``
            A ``torch.FloatTensor``  of shape (batch_size, beam_size),
            Log probabilities of k top sequences.
        """
        if self.attention is not None:
            assert encoder_output is not None

        beam_search = BeamSearch(self.end_index, num_steps, beam_size, per_node_beam_size)
        start_prediction = hidden.new_full((hidden.size(1),), fill_value=self.start_index).long()
        cell_state = hidden.new_zeros((self.num_layers, hidden.size(1), self.hidden_size))

        # `hidden` of shape: (batch_size, num_layers, hidden_size)
        hidden = hidden.transpose(0, 1).contiguous()
        cell_state = cell_state.transpose(0, 1).contiguous()
        state = {'hidden': hidden, 'cell_state': cell_state}
        if self.attention:
            state['encoder_output'] = encoder_output
            state['encoder_mask'] = encoder_mask
        all_top_k_predictions, log_probabilities = \
            beam_search.search(start_prediction, state, self._beam_step, early_stop=early_stop)
        return all_top_k_predictions, log_probabilities

    def _beam_step(self, input, state):
        # shape: (group_size, input_size)
        embedded_input = self.embedding(input)
        rnn_input = embedded_input

        # shape: (group_size, num_layers, input_size)
        hidden = state['hidden']
        cell_state = state['cell_state']
        # shape: (num_layers, group_size, input_size)
        hidden = hidden.transpose(0, 1).contiguous()
        cell_state = cell_state.transpose(0, 1).contiguous()
        if self.attention is not None:
            encoder_output = state['encoder_output']
            encoder_mask = state['encoder_mask']
            # shape: (group_size, num_rows)
            attn_score = self.attention(hidden[-1], encoder_output, encoder_mask)
            attn_input = attn_score.unsqueeze(1).matmul(encoder_output).squeeze(1)
            # shape: (group_size, input_size + attn_size)
            rnn_input = torch.cat([embedded_input, attn_input], dim=-1)
        _, (next_hidden, next_cell_state) = self.rnn(rnn_input.unsqueeze(1), (hidden, cell_state))
        state['hidden'] = next_hidden.transpose(0, 1).contiguous()
        state['cell_state'] = next_cell_state.transpose(0, 1).contiguous()
        # shape: (group_size, vocab_size)
        log_prob = F.log_softmax(self.output_layer(next_hidden[-1]), dim=-1)
        return log_prob, state
