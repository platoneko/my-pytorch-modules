import torch
import torch.nn as nn


class RNNDecoder(nn.Module):
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
                 num_layers=1,
                 dropout=0.0,
                 num_steps=50):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.start_index = start_index
        self.embedder = embedder
        self.attention = attention
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_steps = num_steps
        self.rnn_input_size = self.input_size

        if self.attention is not None:
            self.rnn_input_size += self.attention.value_size

        self.rnn = nn.GRU(input_size=self.rnn_input_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          dropout=self.dropout if self.num_layers > 1 else 0,
                          batch_first=True)

        self.output_layer = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_size, self.num_classes),
            nn.Softmax(dim=-1),
        )

    def forward(self, hidden, target=None, attn_value=None, attn_mask=None,
                train=False, teaching_force_rate=0.0):
        """
        forward
        :param
            hidden (tensor): of shape (batch_size, num_layers, hidden_size)
            target (tensor): target tensor of shape (batch_size, length)
            attn_value (tensor): of shape (batch_size, num_rows, value_size)
            attn_mask (tensor): of shape (batch_size, num_rows)
            train (bool): train stage or not
            teaching_force_rate (float)
        :returns
            probabilities (tensor): of shape (batch_size, num_steps, num_classes)
            predictions (tensor): of shape (batch_size, num_steps)
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
            # prob of shape (batch_size, 1, num_classes)
            prob, hidden = self._take_step(inputs, hidden, attn_value, attn_mask)
            # shape: (batch_size, 1)
            pred = torch.argmax(prob, dim=-1)
            step_probabilities.append(prob)
            step_predictions.append(pred)
            last_predictions = pred.unsqueeze(1)

        probabilities = torch.cat(step_probabilities, dim=1)
        predictions = torch.cat(step_predictions, dim=1)
        return probabilities, predictions

    def _take_step(self, inputs, hidden, attn_value=None, attn_mask=None):
        # shape: (batch_size, 1, input_size)
        embedded_inputs = self.embedder(inputs)
        rnn_inputs = embedded_inputs
        if self.attention is not None:
            # shape: (batch_size, num_rows)
            attn_score = self.attention(hidden, attn_value, attn_mask)
            attn_inputs = torch.sum(attn_score * attn_value, -1).unsqueeze(1)
            # shape: (batch_size, 1, input_size + attn_size)
            rnn_inputs = torch.cat([embedded_inputs, attn_inputs], dim=-1)
        outputs, next_hidden = self.rnn(rnn_inputs, hidden)
        # shape: (batch_size, 1, num_classes)
        prob = self.output_layer(outputs)
        return prob, next_hidden
