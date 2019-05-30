import torch


def masked_softmax(vector, mask, dim=-1,
                   mask_fill_value=-1e32):
    """
    :param
        vector (tensor): of shape (B, *, N)
        mask (tensor): of shape (B, *, N)
    :return
        result (tensor): of shape (B, *, N)
    """
    mask = mask.float()
    masked_vector = vector.masked_fill((1 - mask).byte(), mask_fill_value)
    result = torch.nn.functional.softmax(masked_vector, dim=dim)
    return result
