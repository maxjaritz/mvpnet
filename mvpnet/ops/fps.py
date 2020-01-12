import torch
from . import fps_cuda


class FarthestPointSampleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, num_centroids):
        index = fps_cuda.farthest_point_sample(points, num_centroids)
        return index

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None,) * len(grad_outputs)


def farthest_point_sample(points, num_centroids, transpose=True):
    """Farthest point sample

    Args:
        points (torch.Tensor): (batch_size, 3, num_points)
        num_centroids (int): the number of centroids to sample
        transpose (bool): whether to transpose points

    Returns:
        index (torch.Tensor): (batch_size, num_centroids), sampled indices of centroids.

    """
    if transpose:
        points = points.transpose(1, 2)
    points = points.contiguous()
    return FarthestPointSampleFunction.apply(points, num_centroids)
