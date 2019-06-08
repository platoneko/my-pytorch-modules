"""
:inputs
    query : ``torch.FloatTensor``, required.
        A ``torch.FloatTensor`` of shape (batch, query_size)
    value : ``torch.FloatTensor``, required.
        A ``torch.FloatTensor`` of shape (batch, num_rows, value_size)
    mask : ``torch.LongTensor``, required.
        A ``torch.LongTensor`` of shape (batch, num_rows)
:outputs
    score : ``torch.FloatTensor``
        Normalized attention score, a ``torch.FloatTensor`` of shape (batch, num_rows)
"""

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from utils import masked_softmax


class BilinearAttention(nn.Module):
    """
    score = tanh(x^T W y + b)
    """

    def __init__(self, query_size, value_size):
        super().__init__()
        self._weight_matrix = Parameter(torch.Tensor(query_size, value_size))
        self._bias = Parameter(torch.Tensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self._weight_matrix)
        self._bias.data.fill_(0)

    def forward(self, query, value, mask):
        intermediate = query.mm(self._weight_matrix).unsqueeze(1)
        intermediate = torch.tanh(intermediate.bmm(value.transpose(1, 2)).squeeze(1) + self._bias)
        score = masked_softmax(intermediate, mask)
        return score


class CosineAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, value, mask):
        q_norm = query / (query.norm(p=2, dim=-1, keepdim=True) + 1e-13)
        v_norm = value / (value.norm(p=2, dim=-1, keepdim=True) + 1e-13)
        intermediate = torch.bmm(q_norm.unsqueeze(1), v_norm.transpose(1, 2)).squeeze(1)
        score = masked_softmax(intermediate, mask)
        return score


class DotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, value, mask):
        intermediate = value.bmm(query.unsqueeze(-1)).squeeze(-1)
        score = masked_softmax(intermediate, mask)
        return score


class MLPAttention(nn.Module):
    """
    score = tanh(x^T W1 + y^T W2 + b) W
    """

    def __init__(self, query_size, value_size, hidden_size):
        super().__init__()
        self.linear_query = nn.Linear(query_size, hidden_size, bias=True)
        self.linear_value = nn.Linear(value_size, hidden_size, bias=False)
        self.tanh = nn.Tanh()
        self.linear_hidden = nn.Linear(self.hidden_size, 1, bias=False)
        
    def forward(self, query, value, mask):
        hidden = self.linear_query(query).unsqueeze(1) \
                 + self.linear_memory(value)
        intermediate = self.tanh(hidden)
        intermediate = self.linear_hidden(intermediate).squeeze(-1)
        score = masked_softmax(intermediate, mask)
        return score
