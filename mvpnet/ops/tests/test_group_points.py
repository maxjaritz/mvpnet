import pytest
import torch
from mvpnet.ops.group_points import group_points


def group_points_torch(feature, index):
    """built-in operators"""
    b, c, n1 = feature.size()
    _, n2, k = index.size()
    feature_expand = feature.unsqueeze(2).expand(b, c, n2, n1)
    index_expand = index.unsqueeze(1).expand(b, c, n2, k)
    return torch.gather(feature_expand, 3, index_expand)


test_data = [
    (2, 3, 512, 128, 32, False),
    (5, 64, 513, 129, 33, False),
    (32, 32, 1024, 512, 64, True),
]


@pytest.mark.parametrize('b, c, n1, n2, k, profile', test_data)
def test(b, c, n1, n2, k, profile):
    torch.manual_seed(0)

    feature = torch.randn(b, c, n1).cuda()
    index = torch.randint(0, n1, [b, n2, k]).long().cuda()

    feature_gather = feature.clone()
    feature_gather.requires_grad = True
    feature_cuda = feature.clone()
    feature_cuda.requires_grad = True

    # Check forward
    out_gather = group_points_torch(feature_gather, index)
    out_cuda = group_points(feature_cuda, index)
    assert out_gather.allclose(out_cuda)

    # Check backward
    out_gather.backward(torch.ones_like(out_gather))
    out_cuda.backward(torch.ones_like(out_cuda))
    grad_gather = feature_gather.grad
    grad_cuda = feature_cuda.grad
    assert grad_gather.allclose(grad_cuda)

    if profile:
        with torch.autograd.profiler.profile(use_cuda=torch.cuda.is_available()) as prof:
            out_cuda = group_points(feature_cuda, index)
        print(prof)
        with torch.autograd.profiler.profile(use_cuda=torch.cuda.is_available()) as prof:
            out_cuda.backward(torch.ones_like(out_cuda))
        print(prof)
