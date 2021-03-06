import torch
import numpy as np
import math


def get_sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.

    :param lengths: ``torch.LongTensor``, required.
        A ``torch.LongTensor`` of shape (B,) which contains a batch of sequence lengths.
    :param max_len: ``int``, optional (default = None)
    :return:
        mask : ``torch.ByteTensor``
        A ``torch.ByteTensor`` of shape (B, len)
    """

    if max_len is None:
        max_len = lengths.max().item()
    mask = torch.arange(0, max_len, dtype=torch.long).type_as(lengths)
    mask = mask.unsqueeze(0)
    mask = mask.repeat(1, *lengths.size(), 1)
    mask = mask.squeeze(0)
    mask = mask.lt(lengths.unsqueeze(-1))
    #mask = mask.repeat(*lengths.size(), 1).lt(lengths.unsqueeze(-1))
    return mask


def masked_softmax(vector, mask, dim=-1,
                   mask_fill_value=-1e32):
    """

    :param vector: ``torch.Tensor``, required.
        Shape: (B, *, N)
    :param mask: ``torch.LongTensor``, required.
        Shape: (B, *, N)
    :param dim: ``int``
        The dimension to calculate softmax.
    :param mask_fill_value: ``float``
        Replace mask position with this value.
    """

    masked_vector = vector.masked_fill((1 - mask).byte(), mask_fill_value)
    result = torch.nn.functional.softmax(masked_vector, dim=dim)
    return result


def masked_max(vector, mask, dim, keepdim=False, mask_fill_value=-1e32):
    """
    To calculate max along certain dimensions on masked values

    :param vector: ``torch.Tensor``
        The vector to calculate max.
    :param mask: ``torch.Tensor``
        The mask of the vector. It must be broadcastable with vector.
    :param dim: ``int``
        The dimension to calculate mean.
    :param keepdim: ``bool``
        Whether to keep dimension.
    :param mask_fill_value: ``float``
        Replace mask position with this value.
    :return:
        A ``torch.Tensor`` of including the max values.
    """

    one_minus_mask = (1.0 - mask).byte()
    replaced_vector = vector.masked_fill(one_minus_mask, mask_fill_value)
    value_sum, _ = torch.max(replaced_vector, dim=dim, keepdim=keepdim)
    return value_sum


def masked_sum(vector, mask, dim, keepdim=False):
    """
    To calculate sum along certain dimensions on masked values

    :param vector: ``torch.Tensor``
        The vector to calculate sum.
    :param mask: ``torch.Tensor``
        The mask of the vector. It must be broadcastable with vector.
    :param dim: ``int``
        The dimension to calculate sum.
    :param keepdim: ``bool``
        Whether to keep dimension.
    :return:
        A ``torch.Tensor`` of including the sum values.
    """

    one_minus_mask = (1.0 - mask).byte()
    replaced_vector = vector.masked_fill(one_minus_mask, 0.0)
    value_sum = torch.sum(replaced_vector, dim=dim, keepdim=keepdim)
    return value_sum


def masked_mean(vector, mask, dim, keepdim=False, eps=1e-8):
    """
    To calculate mean along certain dimensions on masked values

    :param vector: ``torch.Tensor``
        The vector to calculate mean.
    :param mask: ``torch.Tensor``
        The mask of the vector. It must be broadcastable with vector.
    :param dim: ``int``
        The dimension to calculate mean.
    :param keepdim: ``bool``
        Whether to keep dimension.
    :param eps: ``float``
        A small value to avoid zero division problem.
    :return:
        A ``torch.Tensor`` of including the mean values.
    """

    one_minus_mask = (1.0 - mask).byte()
    replaced_vector = vector.masked_fill(one_minus_mask, 0.0)
    value_sum = torch.sum(replaced_vector, dim=dim, keepdim=keepdim)
    value_count = torch.sum(mask.float(), dim=dim, keepdim=keepdim)
    return value_sum / value_count.clamp(min=eps)


def sequence_cross_entropy_with_logits(
        logits,
        targets,
        weights,
        average="batch",
        label_smoothing=None
):
    """
    Computes the cross entropy loss of a sequence, weighted with respect to
    some user provided weights. Note that the weighting here is not the same as
    in the :func:`torch.nn.CrossEntropyLoss()` criterion, which is weighting
    classes; here we are weighting the loss contribution from particular elements
    in the sequence. This allows loss computations for models which use padding.

    :param logits: ``torch.FloatTensor``, required.
        A ``torch.FloatTensor`` of shape (batch_size, sequence_length, num_classes)
        which contains the unnormalized probability for each class.
    :param targets: ``torch.LongTensor``, required.
        A ``torch.LongTensor`` of size (batch, sequence_length) which contains the
        index of the true class for each corresponding step.
    :param weights: ``torch.FloatTensor``, required.
        A ``torch.FloatTensor`` of size (batch, sequence_length)
    :param average: str, optional (default = "batch")
        If "batch", average the loss across the batches. If "token", average
        the loss across each item in the input. If ``None``, return a vector
        of losses per batch element.
    :param label_smoothing: ``float``, optional (default = None)
        Whether or not to apply label smoothing to the cross-entropy loss.
        For example, with a label smoothing value of 0.2, a 4 class classification
        target would look like ``[0.05, 0.05, 0.85, 0.05]`` if the 3rd class was
        the correct label.
    :return:
        A torch.FloatTensor representing the cross entropy loss.
        If ``average=="batch"`` or ``average=="token"``, the returned loss is a scalar.
        If ``average is None``, the returned loss is a vector of shape (batch_size,).
    """

    if average not in {None, "token", "batch"}:
        raise ValueError("Got average f{average}, expected one of "
                         "None, 'token', or 'batch'")

    # shape : (batch * sequence_length, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # shape : (batch * sequence_length, num_classes)
    log_probs_flat = torch.nn.functional.log_softmax(logits_flat, dim=-1)
    # shape : (batch * max_len, 1)
    targets_flat = targets.view(-1, 1).long()

    if label_smoothing is not None and label_smoothing > 0.0:
        num_classes = logits.size(-1)
        smoothing_value = label_smoothing / num_classes
        # Fill all the correct indices with 1 - smoothing value.
        one_hot_targets = torch.zeros_like(log_probs_flat).scatter_(-1, targets_flat, 1.0 - label_smoothing)
        smoothed_targets = one_hot_targets + smoothing_value
        negative_log_likelihood_flat = - log_probs_flat * smoothed_targets
        negative_log_likelihood_flat = negative_log_likelihood_flat.sum(-1, keepdim=True)
    else:
        # Contribution to the negative log likelihood only comes from the exact indices
        # of the targets, as the target distributions are one-hot. Here we use torch.gather
        # to extract the indices of the num_classes dimension which contribute to the loss.
        # shape : (batch * sequence_length, 1)
        negative_log_likelihood_flat = - torch.gather(log_probs_flat, dim=1, index=targets_flat)
    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood_flat.view(*targets.size())
    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood * weights.float()

    if average == "batch":
        # shape : (batch_size,)
        per_batch_loss = negative_log_likelihood.sum(1) / (weights.sum(1).float() + 1e-13)
        num_non_empty_sequences = ((weights.sum(1) > 0).float().sum() + 1e-13)
        return per_batch_loss.sum() / num_non_empty_sequences
    elif average == "token":
        return negative_log_likelihood.sum() / (weights.sum().float() + 1e-13)
    else:
        # shape : (batch_size,)
        per_batch_loss = negative_log_likelihood.sum(1) / (weights.sum(1).float() + 1e-13)
        return per_batch_loss


def get_range_vector(size, device):
    """
    Returns a range vector with the desired size, starting at 0. The CUDA implementation
    is meant to avoid copy data from CPU to GPU.
    """
    if device > -1:
        return torch.cuda.LongTensor(size, device=device).fill_(1).cumsum(0) - 1
    else:
        return torch.arange(0, size, dtype=torch.long)


def get_device_of(tensor):
    """
    Returns the device of the tensor.
    """
    if not tensor.is_cuda:
        return -1
    else:
        return tensor.get_device()


def create_position_embedding(n_pos, dim, out):
    position_enc = np.array([
        [pos / np.power(10000, 2 * j / dim) for j in range(dim // 2)]
        for pos in range(n_pos)
    ])

    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc))
    out.detach_()
    out.requires_grad = False
