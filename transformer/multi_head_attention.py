import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, dim, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        assert dim % num_heads == 0

        self.attn_dropout = nn.Dropout(p=dropout)  # --attention-dropout
        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(dim, dim)
        self.v_lin = nn.Linear(dim, dim)
        # TODO: merge for the initialization step
        nn.init.xavier_normal_(self.q_lin.weight)
        nn.init.xavier_normal_(self.k_lin.weight)
        nn.init.xavier_normal_(self.v_lin.weight)
        # and set biases to 0
        self.out_lin = nn.Linear(dim, dim)

        nn.init.xavier_normal_(self.out_lin.weight)

    def forward(self, query, key, value, mask=None):
        """
        forward

        :param
        query : ``torch.FloatTensor``, required.
            A ``torch.FloatTensor`` of shape (batch_size, query_len, dim)
        key : ``torch.FloatTensor``, required.
            A ``torch.FloatTensor`` of shape (batch_size, key_len, dim)
        value : ``torch.FloatTensor``, required.
            A ``torch.FloatTensor`` of shape (batch_size, value_len, dim), (key_len == value_len)
        mask : ``torch.LongTensor``, optional (default = None)
            A ``torch.LongTensor`` of shape (batch_size, key_len) (self-attention) or
            (batch_size, query_len, key_len) (enc-attention)
        :return
        outputs : ``torch.FloatTensor``
            A ``torch.FloatTensor`` of shape (batch_size, query_len, dim)
        """

        batch_size, query_len, dim = query.size()
        assert dim == self.dim, \
            f'Dimensions do not match: {dim} query vs {self.dim} configured'
        num_heads = self.num_heads

        dim_per_head = dim // num_heads
        scale = math.sqrt(dim_per_head)

        def prepare_head(tensor):
            # input is of shape (batch_size, seq_len, num_heads * dim_per_head)
            # output is of shape (batch_size * num_heads, seq_len, dim_per_head)
            seq_len = tensor.size(1)
            tensor = tensor.view(batch_size, seq_len, num_heads, dim_per_head)
            tensor = tensor.transpose(1, 2).contiguous().view(
                batch_size * num_heads,
                seq_len,
                dim_per_head
            )
            return tensor

        _, key_len, dim = key.size()

        q = prepare_head(self.q_lin(query))
        k = prepare_head(self.k_lin(key))
        v = prepare_head(self.v_lin(value))

        # shape: (batch_size * num_heads, query_length, key_length)
        dot_prod = q.bmm(k.transpose(1, 2))
        attn_mask = (mask == 0).\
            view(batch_size, 1, -1, key_len).\
            repeat(1, num_heads, 1, 1).\
            expand(batch_size, num_heads, query_len, key_len).\
            view(batch_size * num_heads, query_len, key_len)
        assert attn_mask.shape == dot_prod.shape
        dot_prod.masked_fill_(attn_mask, -float(1e20))

        attn_score = F.softmax(dot_prod / scale, dim=-1)
        attn_score = self.attn_dropout(attn_score)  # --attention-dropout

        outputs = attn_score.bmm(v)
        outputs = outputs.\
            view(batch_size, num_heads, query_len, dim_per_head).\
            transpose(1, 2).\
            contiguous().\
            view(batch_size, query_len, dim)

        outputs = self.out_lin(outputs)

        return outputs