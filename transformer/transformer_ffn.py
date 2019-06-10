import torch.nn as nn
import torch.nn.functional as F


class TransformerFFN(nn.Module):
    def __init__(self, model_size, hidden_size, dropout=0.0):
        super(TransformerFFN, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.lin1 = nn.Linear(model_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, model_size)
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.xavier_uniform_(self.lin2.weight)
        # TODO: initialize biases to 0

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.dropout(x)  # --relu-dropout
        x = self.lin2(x)
        return x
