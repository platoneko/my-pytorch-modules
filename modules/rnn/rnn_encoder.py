import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


def _bridge_bidirectional_hidden(hidden):
    """
    the bidirectional hidden is (num_layers * num_directions, batch_size, hidden_size)
    we need to convert it to (num_layers, batch_size, num_directions * hidden_size)
    """

    num_layers = hidden.size(0) // 2
    _, batch_size, hidden_size = hidden.size()
    return hidden.view(num_layers, 2, batch_size, hidden_size) \
        .transpose(1, 2).contiguous().view(num_layers, batch_size, hidden_size * 2)


class GRUEncoder(nn.Module):
    """
    A GRU recurrent neural network encoder.
    """

    def __init__(
            self,
            input_size,
            hidden_size,
            embedding,
            num_layers=1,
            bidirectional=True,
            dropout=0.0
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout

        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

    def forward(self, input, hidden=None):
        """
        forward

        :param
        input : ``torch.LongTensor`` or ``Tuple(torch.LongTensor, torch.LongTensor)``, required.
            Input tensor of shape (batch_size, length), or tuple containing input tensor and length.
        hidden : ``torch.FloatTensor``
            A ``torch.FloatTensor`` of shape (num_layers * num_directions, batch, hidden_size),
            containing the initial hidden state for each element in the batch.

        :return
        output : ``torch.FloatTensor``
            A ``torch.FloatTensor`` of shape (batch, length, hidden_size * num_directions).
        last_hidden : : ``torch.FloatTensor``, optional (default = None)
            A ``torch.FloatTensor`` of shape (num_layers, batch, num_directions * hidden_size).
        """
        if isinstance(input, tuple):
            input, length = input
        else:
            input, length = input, None

        batch_size = input.size(0)

        if length is not None:
            #  length may be 0, however pytorch `pack_padded_sequence` can't haddle this case.
            num_valid = length.gt(0).int().sum().item()
            sorted_length, indices = length.sort(descending=True)
            rnn_input = self.embedding(input.index_select(0, indices))

            rnn_input = pack_padded_sequence(
                rnn_input[:num_valid],
                sorted_length[:num_valid].tolist(),
                batch_first=True
            )

            if hidden is not None:
                hidden = hidden.index_select(1, indices)[:, :num_valid]
        else:
            rnn_input = self.embedding(input)

        output, last_hidden = self.rnn(rnn_input, hidden)

        if self.bidirectional:
            last_hidden = _bridge_bidirectional_hidden(last_hidden)

        if length is not None:
            output, _ = pad_packed_sequence(output, batch_first=True)
            if num_valid < batch_size:
                zeros = output.new_zeros(
                    batch_size - num_valid, output.size(1), output.size(2))
                output = torch.cat([output, zeros], dim=0)

                zeros = last_hidden.new_zeros(
                    self.num_layers, batch_size - num_valid, output.size(2))
                last_hidden = torch.cat([last_hidden, zeros], dim=1)
            _, inv_indices = indices.sort()
            output = output.index_select(0, inv_indices)
            last_hidden = last_hidden.index_select(1, inv_indices)

        return output, last_hidden


class LSTMEncoder(nn.Module):
    """
    A LSTM recurrent neural network encoder.
    """

    def __init__(
            self,
            input_size,
            hidden_size,
            embedding,
            num_layers=1,
            bidirectional=True,
            dropout=0.0
    ):
        super(LSTMEncoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

    def forward(self, input, hidden_tuple=None):
        """
        forward

        :param
        input : ``torch.LongTensor`` or ``Tuple(torch.LongTensor, torch.LongTensor)``, required.
            Input tensor of shape (batch_size, length), or tuple containing input tensor and length.
        hidden_tuple : ``Tuple(torch.FloatTensor, torch.FloatTensor)``, optional (default = None)
            tuple of 2 tensor of shape (num_layers * num_directions, batch, hidden_size),
            containing the initial hidden state and initial cell state for each element in the batch.

        :returns
        output : ``torch.FloatTensor``
            A ``torch.FloatTensor`` of shape (batch, length, hidden_size * num_directions).
        last_hidden : : ``torch.FloatTensor``
            A ``torch.FloatTensor`` of shape (num_layers, batch, num_directions * hidden_size).
        last_cell_state : ``torch.FloatTensor``
            A ``torch.FloatTensor`` of shape (num_layers, batch, num_directions * hidden_size).
        """
        if isinstance(input, tuple):
            input, length = input
        else:
            input, length = input, None

        batch_size = input.size(0)

        if length is not None:
            #  length may be 0, however pytorch `pack_padded_sequence` can't haddle this case
            num_valid = length.gt(0).int().sum().item()
            sorted_length, indices = length.sort(descending=True)
            rnn_input = self.embedding(input.index_select(0, indices))

            rnn_input = pack_padded_sequence(
                rnn_input[:num_valid],
                sorted_length[:num_valid].tolist(),
                batch_first=True
            )

            if hidden_tuple is not None:
                hidden, cell_state = hidden_tuple
                hidden = hidden.index_select(1, indices)[:, :num_valid]
                cell_state = cell_state.index_select(1, indices)[:, :num_valid]
                hidden_tuple = (hidden, cell_state)
        else:
            rnn_input = input

        output, (last_hidden, last_cell_state) = self.rnn(rnn_input, hidden_tuple)

        if self.bidirectional:
            last_hidden = _bridge_bidirectional_hidden(last_hidden)
            last_cell_state = _bridge_bidirectional_hidden(last_cell_state)

        if length is not None:
            output, _ = pad_packed_sequence(output, batch_first=True)

            if num_valid < batch_size:
                zeros = output.new_zeros(
                    batch_size - num_valid, output.size(1), output.size(2))
                output = torch.cat([output, zeros], dim=0)

                zeros = last_hidden.new_zeros(
                    self.num_layers, batch_size - num_valid, output.size(2))
                last_hidden = torch.cat([last_hidden, zeros], dim=1)

                zeros = last_cell_state.new_zeros(
                    self.num_layers, batch_size - num_valid, output.size(2))
                last_cell_state = torch.cat([last_cell_state, zeros], dim=1)

            _, inv_indices = indices.sort()
            output = output.index_select(0, inv_indices)
            last_hidden = last_hidden.index_select(1, inv_indices)
            last_cell_state = last_cell_state.index_select(1, inv_indices)

        return output, (last_hidden, last_cell_state)
