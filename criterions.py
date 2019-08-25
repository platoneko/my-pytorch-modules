import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class SequenceNLLLoss(_Loss):
    """
    NLLLoss for sequence, average/sum the loss across the batches
    """
    def __init__(self, weight=None, reduction='mean'):
        super().__init__()
        assert reduction in ['none', 'sum', 'mean']
        self.register_buffer('weight', weight)
        self.reduction = reduction

    def forward(self, inputs, targets, reduction=True):
        """
        inputs: (batch_size, max_len, vocab_size)
        targets: (batch_size, max_len)
        """
        batch_size = inputs.size(0)
        nll = F.nll_loss(
            input=inputs.reshape(-1, inputs.size(-1)),
            target=targets.reshape(-1),
            weight=self.weight,
            reduction='none'
        )
        nll = nll.view(batch_size, -1).sum(dim=1)

        if reduction:
            if self.reduction == 'mean':
                if self.padding_idx is not None:
                    word_cnt = targets.ne(self.padding_idx).float().sum(dim=1)
                    nll = nll / word_cnt
                    nll = nll.mean()
            elif self.reduction == 'sum':
                nll = nll.sum()

        return nll


class SequenceCrossEntropy(_Loss):
    """
    Cross entropy for sequence, average/sum the loss across the batches
    """
    def __init__(self, weight=None, padding_idx=None, reduction='mean'):
        super().__init__()
        assert reduction in ['none', 'sum', 'mean']
        self.register_buffer('weight', weight)
        self.padding_idx = padding_idx
        self.reduction = reduction

    def forward(self, inputs, targets, reduction=True):
        """
        inputs: (batch_size, max_len, vocab_size)
        targets: (batch_size, max_len)
        """
        batch_size = inputs.size(0)
        cross_entropy = F.cross_entropy(
            input=inputs.reshape(-1, inputs.size(-1)),
            target=targets.reshape(-1),
            weight=self.weight,
            reduction='none'
        )
        cross_entropy = cross_entropy.view(batch_size, -1).sum(dim=1)

        if reduction:
            if self.reduction == 'mean':
                if self.padding_idx is not None:
                    word_cnt = targets.ne(self.padding_idx).float().sum(dim=1)
                    cross_entropy = cross_entropy / word_cnt
                    cross_entropy = cross_entropy.mean()
            elif self.reduction == 'sum':
                cross_entropy = cross_entropy.sum()

        return cross_entropy


class FocalLoss(_Loss):
    def __init__(self, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        nll_loss = F.nll_loss(input, target, reduction=self.reduction)
        pt = torch.exp(-nll_loss)
        f_loss = (1 - pt)**self.gamma * nll_loss
        return f_loss
