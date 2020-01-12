import torch
from . import interpolate_cuda


class FeatureInterpolate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feature, index, weight):
        b, c, n = feature.size()
        ctx.save_for_backward(index, weight)
        ctx.n = n
        interpolated_feature = interpolate_cuda.interpolate_forward(feature, index, weight)
        return interpolated_feature

    @staticmethod
    def backward(ctx, *grad_out):
        index, weight = ctx.saved_tensors
        n = ctx.n
        grad_input = interpolate_cuda.interpolate_backward(grad_out[0], index, weight, n)
        return grad_input, None, None


def feature_interpolate(feature, index, weight):
    """Feature interpolate

    Args:
       feature: (B, C, N1), features of key points
       index: (B, N2, K), indices of key points to interpolate
       weight: (b, N2, K), weights to interpolate

    Returns:
       interpolated_feature: (B, C, N2)

    """
    return FeatureInterpolate.apply(feature, index, weight)
