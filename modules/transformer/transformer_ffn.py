import torch.nn as nn
import torch.nn.functional as F


class TransformerFFN(nn.Module):
    def __init__(self, model_dim, hidden_size, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.lin1 = nn.Linear(model_dim, hidden_size)
        self.lin2 = nn.Linear(hidden_size, model_dim)
        self.norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        inter = self.dropout(F.relu(self.lin1(self.norm(x))))
        output = self.dropout(self.lin2(inter))
        return output + x
