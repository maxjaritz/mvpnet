import pytest
import torch
from mvpnet.ops.interpolate import feature_interpolate


def group_points_torch(feature, index):
    """built-in operators"""
    b, c, n1 = feature.size()
    _, n2, k = index.size()
    feature_expand = feature.unsqueeze(2).expand(b, c, n2, n1)
    index_expand = index.unsqueeze(1).expand(b, c, n2, k)
    return torch.gather(feature_expand, 3, index_expand)


def feature_interpolate_torch(feature, index, weight):
    """built-in operators"""
    neighbour_feature = group_points_torch(feature, index)
    weighted_feature = neighbour_feature * weight.unsqueeze(1)
    interpolated_feature = weighted_feature.sum(dim=-1)
    return interpolated_feature


test_data = [
    (2, 64, 128, 512, False),
    (3, 65, 129, 513, False),
    (32, 64, 256, 1024, True),
    (32, 64, 2048, 8192, True),
]


@pytest.mark.parametrize('b, c, n1, n2, profile', test_data)
def test(b, c, n1, n2, profile):
    torch.manual_seed(0)
    k = 3

    feature = torch.randn(b, c, n1).double().cuda()
    index = torch.randint(0, n1, [b, n2, k]).long().cuda()
    weight = torch.rand(b, n2, k).double().cuda()
    weight = weight / weight.sum(dim=2, keepdim=True)

    feature_torch = feature.clone()
    feature_torch.requires_grad = True
    feature_cuda = feature.clone()
    feature_cuda.requires_grad = True

    # Check forward
    out_torch = feature_interpolate_torch(feature_torch, index, weight)
    out_cuda = feature_interpolate(feature_cuda, index, weight)
    assert out_torch.allclose(out_cuda)

    if profile:
        with torch.autograd.profiler.profile(use_cuda=torch.cuda.is_available()) as prof:
            out_cuda = feature_interpolate(feature_cuda, index, weight)
        print(prof)
        with torch.autograd.profiler.profile(use_cuda=torch.cuda.is_available()) as prof:
            out_cuda.backward(torch.ones_like(out_cuda))
        print(prof)
    else:
        # Check backward
        out_torch.backward(torch.ones_like(out_torch))
        out_cuda.backward(torch.ones_like(out_cuda))
        grad_torch = feature_torch.grad
        grad_cuda = feature_cuda.grad
        assert grad_torch.allclose(grad_cuda)
