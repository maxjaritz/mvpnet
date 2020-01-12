import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------- #
# Distance
# ---------------------------------------------------------------------------- #
def bpdist(feature, data_format='NCW'):
    """Compute pairwise (square) distances of features.
    Based on $(x-y)^2=x^2+y^2-2xy$.

    Args:
        feature (torch.Tensor): (batch_size, channels, num_inst)
        data_format (str): the format of features. [NCW/NWC]

    Returns:
        distance (torch.Tensor): (batch_size, num_inst, num_inst)

    Notes:
        This method returns square distances, and is optimized for lower memory and faster speed.
        Square sum is more efficient than gather diagonal from inner product.
        The result is somehow inaccurate compared to directly using $(x-y)^2$.

    """
    assert data_format in ('NCW', 'NWC')
    if data_format == 'NCW':
        square_sum = torch.sum(feature ** 2, 1, keepdim=True)
        square_sum = square_sum.transpose(1, 2) + square_sum
        distance = torch.baddbmm(square_sum, feature.transpose(1, 2), feature, alpha=-2.0)
    else:
        square_sum = torch.sum(feature ** 2, 2, keepdim=True)
        square_sum = square_sum.transpose(1, 2) + square_sum
        distance = torch.baddbmm(square_sum, feature, feature.transpose(1, 2), alpha=-2.0)
    return distance


def bpdist2(feature1, feature2, data_format='NCW'):
    """Compute pairwise (square) distances of features.

    Args:
        feature1 (torch.Tensor): (batch_size, channels, num_inst1)
        feature2 (torch.Tensor): (batch_size, channels, num_inst2)
        data_format (str): the format of features. [NCW/NWC]

    Returns:
        distance (torch.Tensor): (batch_size, num_inst1, num_inst2)

    """
    assert data_format in ('NCW', 'NWC')
    if data_format == 'NCW':
        square_sum1 = torch.sum(feature1 ** 2, 1, keepdim=True)
        square_sum2 = torch.sum(feature2 ** 2, 1, keepdim=True)
        square_sum = square_sum1.transpose(1, 2) + square_sum2
        distance = torch.baddbmm(square_sum, feature1.transpose(1, 2), feature2, alpha=-2.0)
    else:
        square_sum1 = torch.sum(feature1 ** 2, 2, keepdim=True)
        square_sum2 = torch.sum(feature2 ** 2, 2, keepdim=True)
        square_sum = square_sum1 + square_sum2.transpose(1, 2)
        distance = torch.baddbmm(square_sum, feature1, feature2.transpose(1, 2), alpha=-2.0)
    return distance


def pdist2(feature1, feature2):
    """Compute pairwise (square) distances of features.

    Args:
        feature1 (torch.Tensor): (num_inst1, channels)
        feature2 (torch.Tensor): (num_inst2, channels)

    Returns:
        distance (torch.Tensor): (num_inst1, num_inst2)

    """
    square_sum1 = torch.sum(feature1 ** 2, 1, keepdim=True)
    square_sum2 = torch.sum(feature2 ** 2, 1, keepdim=True)
    square_sum = square_sum1 + square_sum2.transpose(0, 1)
    distance = torch.addmm(square_sum, feature1, feature2.transpose(0, 1), alpha=-2.0)
    return distance


# ---------------------------------------------------------------------------- #
# Losses
# ---------------------------------------------------------------------------- #
def encode_one_hot(target, num_classes):
    """Encode integer labels into one-hot vectors

    Args:
        target (torch.Tensor): (N,)
        num_classes (int): the number of classes

    Returns:
        torch.FloatTensor: (N, C)

    """
    one_hot = target.new_zeros(target.size(0), num_classes)
    one_hot = one_hot.scatter(1, target.unsqueeze(1), 1)
    return one_hot.float()


def smooth_cross_entropy(input, target, label_smoothing):
    """Cross entropy loss with label smoothing

    Args:
        input (torch.Tensor): (N, C)
        target (torch.Tensor): (N,)
        label_smoothing (float):

    Returns:
        loss (torch.Tensor): scalar

    """
    assert input.dim() == 2 and target.dim() == 1
    assert isinstance(label_smoothing, float)
    batch_size, num_classes = input.shape
    one_hot = torch.zeros_like(input).scatter(1, target.unsqueeze(1), 1)
    smooth_one_hot = one_hot * (1 - label_smoothing) + torch.ones_like(input) * (label_smoothing / num_classes)
    log_prob = F.log_softmax(input, dim=1)
    loss = (- smooth_one_hot * log_prob).sum(1).mean()
    return loss


# ---------------------------------------------------------------------------- #
# Indexing
# ---------------------------------------------------------------------------- #
def batch_index_select(input, index, dim):
    """Batch index_select

    References: https://discuss.pytorch.org/t/batched-index-select/9115/7

    Args:
        input (torch.Tensor): (b, ...)
        index (torch.Tensor): (b, n)
        dim (int): the dimension to index

    """
    assert index.dim() == 2, 'Index should be 2-dim.'
    assert input.size(0) == index.size(0), 'Mismatched batch size: {} vs {}'.format(input.size(0), index.size(0))
    batch_size = index.size(0)
    num_select = index.size(1)
    views = [1 for _ in range(input.dim())]
    views[0] = batch_size
    views[dim] = num_select
    expand_shape = list(input.shape)
    expand_shape[dim] = -1
    index = index.view(views).expand(expand_shape)
    return torch.gather(input, dim, index)
