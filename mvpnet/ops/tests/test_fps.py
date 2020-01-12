import pytest
import numpy as np
import torch
from mvpnet.ops.fps import farthest_point_sample


def farthest_point_sample_np(points: np.ndarray, num_centroids: int, transpose=True) -> np.ndarray:
    """Farthest point sample (numpy version)

    Args:
        points: (batch_size, 3, num_points)
        num_centroids: the number of centroids
        transpose: whether to transpose points

    Returns:
        index: index of centroids. (batch_size, num_centroids)

    """
    if transpose:
        points = np.transpose(points, [0, 2, 1])
    index = []
    for points_per_batch in points:
        index_per_batch = [0]
        cur_ind = 0
        dist2set = None
        for ind in range(1, num_centroids):
            cur_xyz = points_per_batch[cur_ind]
            dist2cur = points_per_batch - cur_xyz[None, :]
            dist2cur = np.square(dist2cur).sum(1)
            if dist2set is None:
                dist2set = dist2cur
            else:
                dist2set = np.minimum(dist2cur, dist2set)
            cur_ind = np.argmax(dist2set)
            index_per_batch.append(cur_ind)
        index.append(index_per_batch)
    return np.asarray(index)


test_data = [
    (2, 3, 1024, 128, True, False),
    (2, 2, 1024, 128, True, False),
    (3, 3, 1025, 129, True, False),
    (3, 3, 1025, 129, False, False),
    (16, 3, 1024, 512, True, True),
    (32, 3, 1024, 512, True, True),
    (32, 3, 8192, 2048, True, True),
]


@pytest.mark.parametrize('batch_size, channels, num_points, num_centroids, transpose, profile', test_data)
def test(batch_size, channels, num_points, num_centroids, transpose, profile):
    np.random.seed(0)
    if transpose:
        points = np.random.rand(batch_size, channels, num_points)
    else:
        points = np.random.rand(batch_size, num_points, channels)

    index_np = farthest_point_sample_np(points, num_centroids, transpose=transpose)
    point_tensor = torch.from_numpy(points).cuda()
    index_tensor = farthest_point_sample(point_tensor, num_centroids, transpose=transpose)
    np.testing.assert_equal(index_np, index_tensor.cpu().numpy())

    if profile:
        with torch.autograd.profiler.profile(use_cuda=torch.cuda.is_available()) as prof:
            farthest_point_sample(point_tensor, num_centroids)
        print(prof)
