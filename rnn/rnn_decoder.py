import torch
import torch.nn as nn
from beam_search import BeamSearch


class GRUDecoder(nn.Module):
    """
    A GRU recurrent neural network decoder.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_classes,
                 start_index,
                 embedder=None,
                 attention=None,
                 num_steps=50):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.start_index = start_index
        self.embedder = embedder
        self.attention = attention
        self.num_steps = num_steps
        self.rnn_input_size = self.input_size

        if self.attention is not None:
            self.rnn_input_size += self.attention.value_size

        self.rnn = nn.GRUCell(input_size=self.rnn_input_size,
                              hidden_size=self.hidden_size)

        self.output_layer = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_size, self.num_classes),
            nn.Softmax(dim=-1),
        )

    def forward(self,
                hidden,
                target=None,
                attn_value=None,
                attn_mask=None,
                train=False,
                teaching_force_rate=0.0):
        """
        forward

        :param
        hidden : ``torch.FloatTensor``, required.
            Initial hidden tensor of shape (batch_size, hidden_size)
        target : ``torch.LongTensor``, optional (default = None)
            Target tokens tensor of shape (batch_size, length)
        attn_value : ``torch.FloatTensor``, optional (default = None)
            A ``torch.FloatTensor`` of shape (batch_size, num_rows, value_size)
        attn_mask : ``torch.LongTensor``, optional (default = None)
            A ``torch.LongTensor`` of shape (batch_size, num_rows)
        train : ``bool``, optional (default = False)
            Train stage or not.
        teaching_force_rate : ``float``, optional (default = 0.0)

        :return
        probabilities : ``torch.FloatTensor``
            A ``torch.FloatTensor`` of shape (batch_size, num_steps, num_classes)
        predictions : ``torch.LongTensor``
            Predicted target tokens tensor of shape (batch_size, num_steps)
        """
        if self.attention is not None:
            assert attn_value is not None

        if train and target is not None:
            # Discard the last eos token in target
            num_steps = target.size(1) - 1
        else:
            num_steps = self.num_steps

        last_predictions = hidden.new_full((hidden.size(0),), fill_value=self.start_index)
        step_probabilities = []
        step_predictions = []
        for timestep in range(num_steps):
            if train and target and torch.rand(1).item() < teaching_force_rate:
                inputs = target[:, timestep]
            else:
                inputs = last_predictions
            # prob of shape (batch_size, num_classes)
            prob, hidden = self._take_step(inputs, hidden, attn_value, attn_mask)
            # shape: (batch_size,)
            last_predictions = torch.argmax(prob, dim=-1)
            step_probabilities.append(prob.unsqueeze(1))
            step_predictions.append(last_predictions.unsqueeze(1))

        probabilities = torch.cat(step_probabilities, dim=1)
        predictions = torch.cat(step_predictions, dim=1)
        return probabilities, predictions

    def _take_step(self, inputs, hidden, attn_value=None, attn_mask=None):
        # shape: (batch_size, input_size)
        embedded_inputs = self.embedder(inputs)
        rnn_inputs = embedded_inputs
        if self.attention is not None:
            # shape: (batch_size, num_rows)
            attn_score = self.attention(hidden, attn_value, attn_mask)
            attn_inputs = torch.sum(attn_score * attn_value, dim=-1)
            # shape: (batch_size, input_size + attn_size)
            rnn_inputs = torch.cat([embedded_inputs, attn_inputs], dim=-1)
        outputs, next_hidden = self.rnn(rnn_inputs, hidden)
        # shape: (batch_size, num_classes)
        prob = self.output_layer(outputs)
        return prob, next_hidden

    def forward_beam_search(self,
                            hidden,
                            end_index,
                            attn_value=None,
                            attn_mask=None,
                            beam_size=4,
                            per_node_beam_size=4):
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

        if beam_size <= 1:
            _, predictions = self.forward(hidden, attn_value=attn_value, attn_mask=attn_mask)
            return predictions

        beam_search = BeamSearch(end_index, self.num_steps, beam_size, per_node_beam_size)
        start_predictions = hidden.new_full((hidden.size(0),), fill_value=self.start_index)

        state = {'hidden': hidden}
        if self.attention:
            state['attn_value'] = attn_value
            state['attn_mask'] = attn_mask
        all_top_k_predictions, log_probabilities = beam_search.search(start_predictions, state, self._beam_step)
        return all_top_k_predictions, log_probabilities

    def _beam_step(self, inputs, state):
        # shape: (group_size, input_size)
        embedded_inputs = self.embedder(inputs)
        rnn_inputs = embedded_inputs
        hidden = state['hidden']
        if self.attention is not None:
            attn_value = state['attn_value']
            attn_mask = state['attn_mask']
            # shape: (group_size, num_rows)
            attn_score = self.attention(hidden, attn_value, attn_mask)
            attn_inputs = torch.sum(attn_score * attn_value, -1).unsqueeze(1)
            # shape: (group_size, input_size + attn_size)
            rnn_inputs = torch.cat([embedded_inputs, attn_inputs], dim=-1)
        outputs, next_hidden = self.rnn(rnn_inputs, hidden)
        state['hidden'] = next_hidden
        # shape: (group_size, num_classes)
        log_prob = torch.log(self.output_layer(outputs))
        return log_prob, state


class LSTMDecoder(nn.Module):
    """
    A LSTM recurrent neural network decoder.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_classes,
                 start_index,
                 embedder=None,
                 attention=None,
                 num_steps=50):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.start_index = start_index
        self.embedder = embedder
        self.attention = attention
        self.num_steps = num_steps
        self.rnn_input_size = self.input_size

        if self.attention is not None:
            self.rnn_input_size += self.attention.value_size

        self.rnn = nn.LSTMCell(input_size=self.rnn_input_size,
                               hidden_size=self.hidden_size)

        self.output_layer = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_size, self.num_classes),
            nn.Softmax(dim=-1),
        )

    def forward(self,
                hidden,
                target=None,
                attn_value=None,
                attn_mask=None,
                train=False,
                teaching_force_rate=0.0):
        """
        forward

        :param
        hidden : ``torch.FloatTensor``, required.
            Initial hidden tensor of shape (batch_size, hidden_size)
        target : ``torch.LongTensor``, optional (default = None)
            Target tokens tensor of shape (batch_size, length)
        attn_value : ``torch.FloatTensor``, optional (default = None)
            A ``torch.FloatTensor`` of shape (batch_size, num_rows, value_size)
        attn_mask : ``torch.LongTensor``, optional (default = None)
            A ``torch.LongTensor`` of shape (batch_size, num_rows)
        train : ``bool``, optional (default = False)
            Train stage or not.
        teaching_force_rate : ``float``, optional (default = 0.0)

        :return
        probabilities : ``torch.FloatTensor``
            A ``torch.FloatTensor`` of shape (batch_size, num_steps, num_classes)
        predictions : ``torch.LongTensor``
            Predicted target tokens tensor of shape (batch_size, num_steps)
        """
        if self.attention is not None:
            assert attn_value is not None

        if train and target is not None:
            # Discard the last eos token in target
            num_steps = target.size(1) - 1
        else:
            num_steps = self.num_steps

        last_predictions = hidden.new_full((hidden.size(0),), fill_value=self.start_index)
        cell_state = hidden.new_full((hidden.size(0), self.hidden_size), fill_value=0.0)
        hidden_tuple = (hidden, cell_state)

        step_probabilities = []
        step_predictions = []
        for timestep in range(num_steps):
            if train and target and torch.rand(1).item() < teaching_force_rate:
                inputs = target[:, timestep]
            else:
                inputs = last_predictions
            # prob of shape (batch_size, num_classes)
            prob, hidden = self._take_step(inputs, hidden_tuple, attn_value, attn_mask)
            # shape: (batch_size,)
            last_predictions = torch.argmax(prob, dim=-1)
            step_probabilities.append(prob.unsqueeze(1))
            step_predictions.append(last_predictions.unsqueeze(1))

        probabilities = torch.cat(step_probabilities, dim=1)
        predictions = torch.cat(step_predictions, dim=1)
        return probabilities, predictions

    def _take_step(self, inputs, hidden_tuple, attn_value=None, attn_mask=None):
        # shape: (batch_size, input_size)
        embedded_inputs = self.embedder(inputs)
        rnn_inputs = embedded_inputs
        if self.attention is not None:
            # shape: (batch_size, num_rows)
            attn_score = self.attention(hidden_tuple[0], attn_value, attn_mask)
            attn_inputs = torch.sum(attn_score * attn_value, dim=-1)
            # shape: (batch_size, input_size + attn_size)
            rnn_inputs = torch.cat([embedded_inputs, attn_inputs], dim=-1)
        outputs, (next_hidden, next_cell_state) = self.rnn(rnn_inputs, hidden_tuple)
        # shape: (batch_size, num_classes)
        prob = self.output_layer(outputs)
        return prob, next_hidden, next_cell_state

    def forward_beam_search(self,
                            hidden,
                            end_index,
                            attn_value=None,
                            attn_mask=None,
                            beam_size=4,
                            per_node_beam_size=4):
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

        if beam_size <= 1:
            _, predictions = self.forward(hidden, attn_value=attn_value, attn_mask=attn_mask)
            return predictions

        beam_search = BeamSearch(end_index, self.num_steps, beam_size, per_node_beam_size)
        start_predictions = hidden.new_full((hidden.size(0),), fill_value=self.start_index)
        cell_state = hidden.new_full((hidden.size(0), self.hidden_size), fill_value=0.0)

        state = {'hidden': hidden, 'cell_state': cell_state}
        if self.attention:
            state['attn_value'] = attn_value
            state['attn_mask'] = attn_mask
        all_top_k_predictions, log_probabilities = beam_search.search(start_predictions, state, self._beam_step)
        return all_top_k_predictions, log_probabilities

    def _beam_step(self, inputs, state):
        # shape: (group_size, input_size)
        embedded_inputs = self.embedder(inputs)
        rnn_inputs = embedded_inputs
        hidden = state['hidden']
        cell_state = state['cell_state']
        if self.attention is not None:
            attn_value = state['attn_value']
            attn_mask = state['attn_mask']
            # shape: (group_size, num_rows)
            attn_score = self.attention(hidden, attn_value, attn_mask)
            attn_inputs = torch.sum(attn_score * attn_value, dim=-1)
            # shape: (group_size, input_size + attn_size)
            rnn_inputs = torch.cat([embedded_inputs, attn_inputs], dim=-1)
        outputs, (next_hidden, next_cell_state) = self.rnn(rnn_inputs, (hidden, cell_state))
        state['hidden'] = next_hidden
        state['cell_state'] = next_cell_state
        # shape: (group_size, num_classes)
        log_prob = torch.log(self.output_layer(outputs))
        return log_prob, state
