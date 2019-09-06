"""
:inputs
    query : ``torch.FloatTensor``, required.
        A ``torch.FloatTensor`` of shape (batch, query_size)
    value : ``torch.FloatTensor``, required.
        A ``torch.FloatTensor`` of shape (batch, value_len, value_size)
    mask : ``torch.LongTensor``, required.
        A ``torch.LongTensor`` of shape (batch, value_len)
:outputs
    score : ``torch.FloatTensor``
        Normalized attention score, a ``torch.FloatTensor`` of shape (batch, value_len)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from modules.utils import masked_softmax


class BilinearAttention(nn.Module):
    """
    score = tanh(x^T W y + b)
    """

    def __init__(self, query_size, value_size):
        super().__init__()
        self.query_size = query_size
        self.value_size = value_size
        self._weight_matrix = Parameter(torch.Tensor(query_size, value_size))
        self._bias = Parameter(torch.Tensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self._weight_matrix)
        self._bias.data.fill_(0)

    def forward(self, query, value, mask=None):
        intermediate = query.mm(self._weight_matrix).unsqueeze(1)
        intermediate = torch.tanh(intermediate.bmm(value.transpose(1, 2)).squeeze(1) + self._bias)
        if mask is not None:
            score = masked_softmax(intermediate, mask)
        else:
            score = F.softmax(intermediate, dim=-1)
        return score


class CosineAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, value, mask=None):
        q_norm = query / (query.norm(p=2, dim=-1, keepdim=True) + 1e-13)
        v_norm = value / (value.norm(p=2, dim=-1, keepdim=True) + 1e-13)
        intermediate = torch.bmm(q_norm.unsqueeze(1), v_norm.transpose(1, 2)).squeeze(1)
        if mask is not None:
            score = masked_softmax(intermediate, mask)
        else:
            score = F.softmax(intermediate, dim=-1)
        return score


class DotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, value, mask=None):
        intermediate = value.bmm(query.unsqueeze(-1)).squeeze(-1)
        if mask is not None:
            score = masked_softmax(intermediate, mask)
        else:
            score = F.softmax(intermediate, dim=-1)
        return score


class MLPAttention(nn.Module):
    """
    score = tanh(x^T W1 + y^T W2 + b) W
    """

    def __init__(self, query_size, value_size, hidden_size):
        super().__init__()
        self.query_size = query_size
        self.value_size = value_size
        self.hidden_size = hidden_size
        self.linear_query = nn.Linear(query_size, hidden_size, bias=True)
        self.linear_value = nn.Linear(value_size, hidden_size, bias=False)
        self.tanh = nn.Tanh()
        self.linear_hidden = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, query, value, mask=None):
        hidden = self.linear_query(query).unsqueeze(1) \
                 + self.linear_value(value)
        intermediate = self.tanh(hidden)
        intermediate = self.linear_hidden(intermediate).squeeze(-1)
        if mask is not None:
            score = masked_softmax(intermediate, mask)
        else:
            score = F.softmax(intermediate, dim=-1)
        return score
