import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.beam_search import BeamSearch


class GRUDecoder(nn.Module):
    """
    A GRU recurrent neural network decoder.
    """
    def __init__(
            self,
            input_size,
            hidden_size,
            start_index,
            end_index,
            embedding,
            output_layer,
            attention=None,
            dropout=0.0
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.start_index = start_index
        self.end_index = end_index
        self.embedding = embedding
        self.attention = attention
        self.dropout = dropout

        self.rnn = nn.GRUCell(self.input_size, self.hidden_size)
        self.output_layer = output_layer

    def forward(
            self,
            hidden,
            target=None,
            attn_value=None,
            attn_mask=None,
            num_steps=50,
            is_training=False
    ):
        """
        forward

        :param
        hidden : ``torch.FloatTensor``, required.
            Initial hidden tensor of shape (batch_size, hidden_size)
        target : ``torch.LongTensor``, optional (default = None)
            Target tokens tensor of shape (batch_size, length), `target` must contain <sos> and <eos> token.
        attn_value : ``torch.FloatTensor``, optional (default = None)
            A ``torch.FloatTensor`` of shape (batch_size, num_rows, value_size)
        attn_mask : ``torch.LongTensor``, optional (default = None)
            A ``torch.LongTensor`` of shape (batch_size, num_rows)
        is_training : ``bool``, optional (default = False).

        :return
        logits : ``torch.FloatTensor``
            A ``torch.FloatTensor`` of shape (batch_size, num_steps, vocab_size)
        """
        if self.attention is not None:
            assert attn_value is not None

        if is_training:
            assert target is not None
            num_steps = target.size(1) - 1

        step_logits = []
        if is_training:
            for timestep in range(num_steps):
                input = target[:, timestep]
                output, hidden = self._take_step(input, hidden, attn_value, attn_mask)
                step_logits.append(output.unsqueeze(1))
        else:
            last_prediction = hidden.new_full((hidden.size(1),), fill_value=self.start_index).long()
            for timestep in range(num_steps):
                if (last_prediction == self.end_index).all():
                    break
                input = last_prediction
                # `output` of shape (batch_size, vocab_size)
                output, hidden = self._take_step(input, hidden, attn_value, attn_mask)
                # shape: (batch_size,)
                last_prediction = torch.argmax(output, dim=-1)
                step_logits.append(output.unsqueeze(1))

        logits = torch.cat(step_logits, dim=1)
        return logits

    def _take_step(self, input, hidden, attn_value=None, attn_mask=None):
        # shape: (batch_size, input_size)
        embedded_input = self.embedding(input)
        rnn_input = embedded_input
        if self.attention is not None:
            # shape: (batch_size, num_rows)
            attn_score = self.attention(hidden, attn_value, attn_mask)
            attn_input = attn_score.unsqueeze(1).matmul(attn_value).squeeze(1)
            # shape: (batch_size, input_size + attn_size)
            rnn_input = torch.cat([embedded_input, attn_input], dim=-1)
        next_hidden = self.rnn(rnn_input, hidden)
        # shape: (batch_size, vocab_size)
        output = self.output_layer(next_hidden)
        return output, next_hidden

    def forward_beam_search(
            self,
            hidden,
            attn_value=None,
            attn_mask=None,
            num_steps=50,
            beam_size=4,
            per_node_beam_size=4,
    ):
        """
        Decoder forward using beam search at inference stage

        :param
        hidden : ``torch.FloatTensor``, required.
            Initial hidden tensor of shape (batch_size, hidden_size)
        end_index : ``int``, required.
            Vocab index of <eos> token.
        attn_value : ``torch.FloatTensor``, optional (default = None)
            A ``torch.FloatTensor`` of shape (batch_size, num_rows, value_size)
        attn_mask : ``torch.LongTensor``, optional (default = None)
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
            assert attn_value is not None

        beam_search = BeamSearch(self.end_index, num_steps, beam_size, per_node_beam_size)
        start_prediction = hidden.new_full((hidden.size(0),), fill_value=self.start_index).long()

        state = {'hidden': hidden}
        if self.attention:
            state['attn_value'] = attn_value
            state['attn_mask'] = attn_mask
        all_top_k_predictions, log_probabilities = \
            beam_search.search(start_prediction, state, self._beam_step)
        return all_top_k_predictions, log_probabilities

    def _beam_step(self, input, state):
        # shape: (group_size, input_size)
        embedded_input = self.embedding(input)
        rnn_input = embedded_input
        hidden = state['hidden']
        if self.attention is not None:
            attn_value = state['attn_value']
            attn_mask = state['attn_mask']
            # shape: (group_size, num_rows)
            attn_score = self.attention(hidden, attn_value, attn_mask)
            attn_input = attn_score.unsqueeze(1).matmul(attn_value).squeeze(1)
            # shape: (group_size, input_size + attn_size)
            rnn_input = torch.cat([embedded_input, attn_input], dim=-1)
        next_hidden = self.rnn(rnn_input, hidden)
        state['hidden'] = next_hidden
        # shape: (group_size, vocab_size)
        log_prob = F.log_softmax(self.output_layer(next_hidden), dim=-1)
        return log_prob, state


class LSTMDecoder(nn.Module):
    """
    A LSTM recurrent neural network decoder.
    """
    def __init__(
            self,
            input_size,
            hidden_size,
            start_index,
            end_index,
            embedding,
            output_layer,
            attention=None,
            dropout=0.0
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.start_index = start_index
        self.end_index = end_index
        self.embedding = embedding
        self.attention = attention
        self.dropout = dropout

        self.rnn = nn.LSTMCell(self.input_size, self.hidden_size)
        self.output_layer = output_layer

    def forward(
            self,
            hidden,
            target=None,
            attn_value=None,
            attn_mask=None,
            num_steps=50,
            is_training=False
    ):
        """
        forward

        :param
        hidden : ``torch.FloatTensor``, required.
            Initial hidden tensor of shape (batch_size, hidden_size)
        target : ``torch.LongTensor``, optional (default = None)
            Target tokens tensor of shape (batch_size, length), `target` must contain <sos> and <eos> token.
        attn_value : ``torch.FloatTensor``, optional (default = None)
            A ``torch.FloatTensor`` of shape (batch_size, num_rows, value_size)
        attn_mask : ``torch.LongTensor``, optional (default = None)
            A ``torch.LongTensor`` of shape (batch_size, num_rows)
        is_training : ``bool``, optional (default = False).

        :return
        logits : ``torch.FloatTensor``
            A ``torch.FloatTensor`` of shape (batch_size, num_steps, vocab_size)
        """
        if self.attention is not None:
            assert attn_value is not None

        if is_training:
            assert target is not None
            num_steps = target.size(1) - 1

        cell_state = hidden.new_full((hidden.size(0), self.hidden_size), fill_value=0.0)
        hidden_tuple = (hidden, cell_state)

        step_logits = []
        if is_training:
            for timestep in range(num_steps):
                input = target[:, timestep]
                output, hidden = self._take_step(input, hidden_tuple, attn_value, attn_mask)
                step_logits.append(output.unsqueeze(1))
        else:
            last_prediction = hidden.new_full((hidden.size(1),), fill_value=self.start_index).long()
            for timestep in range(num_steps):
                if (last_prediction == self.end_index).all():
                    break
                input = last_prediction
                # `output` of shape (batch_size, vocab_size)
                output, hidden = self._take_step(input, hidden_tuple, attn_value, attn_mask)
                # shape: (batch_size,)
                last_prediction = torch.argmax(output, dim=-1)
                step_logits.append(output.unsqueeze(1))
        logits = torch.cat(step_logits, dim=1)
        return logits

    def _take_step(self, input, hidden_tuple, attn_value=None, attn_mask=None):
        # shape: (batch_size, input_size)
        embedded_input = self.embedding(input)
        rnn_input = embedded_input
        if self.attention is not None:
            # shape: (batch_size, num_rows)
            attn_score = self.attention(hidden_tuple[0], attn_value, attn_mask)
            attn_input = attn_score.unsqueeze(1).matmul(attn_value).squeeze(1)
            # shape: (batch_size, input_size + attn_size)
            rnn_input = torch.cat([embedded_input, attn_input], dim=-1)
        next_hidden, next_cell_state = self.rnn(rnn_input, hidden_tuple)
        # shape: (batch_size, vocab_size)
        output = self.output_layer(next_hidden)
        return output, (next_hidden, next_cell_state)

    def forward_beam_search(
            self,
            hidden,
            attn_value=None,
            attn_mask=None,
            num_steps=50,
            beam_size=4,
            per_node_beam_size=4,
    ):
        """
        Decoder forward using beam search at inference stage

        :param
        hidden : ``torch.FloatTensor``, required.
            Initial hidden tensor of shape (batch_size, hidden_size)
        end_index : ``int``, required.
            Vocab index of <eos> token.
        attn_value : ``torch.FloatTensor``, optional (default = None)
            A ``torch.FloatTensor`` of shape (batch_size, num_rows, value_size)
        attn_mask : ``torch.LongTensor``, optional (default = None)
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
            assert attn_value is not None

        beam_search = BeamSearch(self.end_index, num_steps, beam_size, per_node_beam_size)
        start_prediction = hidden.new_full((hidden.size(0),), fill_value=self.start_index).long()
        cell_state = hidden.new_full((hidden.size(0), self.hidden_size), fill_value=0.0)

        state = {'hidden': hidden, 'cell_state': cell_state}
        if self.attention:
            state['attn_value'] = attn_value
            state['attn_mask'] = attn_mask
        all_top_k_predictions, log_probabilities = \
            beam_search.search(start_prediction, state, self._beam_step)
        return all_top_k_predictions, log_probabilities

    def _beam_step(self, input, state):
        # shape: (group_size, input_size)
        embedded_input = self.embedding(input)
        rnn_input = embedded_input
        hidden = state['hidden']
        cell_state = state['cell_state']
        if self.attention is not None:
            attn_value = state['attn_value']
            attn_mask = state['attn_mask']
            # shape: (group_size, num_rows)
            attn_score = self.attention(hidden, attn_value, attn_mask)
            attn_input = attn_score.unsqueeze(1).matmul(attn_value).squeeze(1)
            # shape: (group_size, input_size + attn_size)
            rnn_input = torch.cat([embedded_input, attn_input], dim=-1)
        next_hidden, next_cell_state = self.rnn(rnn_input, (hidden, cell_state))
        state['hidden'] = next_hidden
        state['cell_state'] = next_cell_state
        # shape: (group_size, vocab_size)
        log_prob = F.log_softmax(self.output_layer(next_hidden), dim=-1)
        return log_prob, state
