import pytest
import numpy as np
import torch
from mvpnet.ops.ball_query import ball_query
from mvpnet.ops.ball_query import ball_query_distance

test_data = [
    (2, 64, 128, 0.1, 32, True, False),
    (3, 65, 129, 0.1, 32, True, False),
    (3, 65, 129, 10.0, 32, True, False),
    (3, 65, 129, 0.1, 32, False, False),
    (32, 512, 1024, 0.1, 64, True, True),
]


def ball_query_np(query, key, radius, max_neighbors, transpose=True):
    index = []
    if transpose:
        query = query.transpose([0, 2, 1])
        key = key.transpose([0, 2, 1])
    n1 = query.shape[1]
    # n2 = key.shape[1]

    for query_per_batch, key_per_batch in zip(query, key):
        index_per_batch = np.full([n1, max_neighbors], -1, dtype=np.int64)
        distance_per_batch = np.full([n1, max_neighbors], -1.0, dtype=np.float32)
        for i in range(n1):
            cur_query = query_per_batch[i]
            dist2cur = key_per_batch - cur_query[None, :]
            dist2cur = np.square(dist2cur).sum(1)
            neighbor_index = np.nonzero(dist2cur < (radius ** 2))[0]
            assert neighbor_index.size > 0
            if neighbor_index.size < max_neighbors:
                index_per_batch[i, :neighbor_index.size] = neighbor_index
                index_per_batch[i, neighbor_index.size:] = neighbor_index[0]
                distance_per_batch[i, :neighbor_index.size] = dist2cur[neighbor_index]
            else:
                index_per_batch[i, :] = neighbor_index[:max_neighbors]
                distance_per_batch[i, :] = dist2cur[neighbor_index[:max_neighbors]]
        index.append(index_per_batch)
    return np.asarray(index)


@pytest.mark.parametrize('b, n1, n2, r, k, transpose, profile', test_data)
def test_ball_query(b, n1, n2, r, k, transpose, profile):
    np.random.seed(0)
    if transpose:
        key = np.random.randn(b, 3, n2)
        query = np.array([p[:, np.random.choice(n2, n1, replace=False)] for p in key])
    else:
        key = np.random.randn(b, n2, 3)
        query = np.array([p[np.random.choice(n2, n1, replace=False)] for p in key])
    # key = key.astype(np.float32)
    # query = query.astype(np.float32)
    index_np = ball_query_np(query, key, r, k, transpose=transpose)
    # index_np = index_np.astype(np.int64)

    query_tensor = torch.tensor(query).cuda()
    key_tensor = torch.tensor(key).cuda()
    index_tensor = ball_query(query_tensor, key_tensor, r, k, transpose=transpose)
    np.testing.assert_equal(index_np, index_tensor.cpu().numpy())

    if profile:
        query_tensor = query_tensor.float()
        key_tensor = key_tensor.float()
        with torch.autograd.profiler.profile(use_cuda=torch.cuda.is_available()) as prof:
            ball_query_distance(query_tensor, key_tensor, r, k, transpose=transpose)
        print(prof)


def ball_query_distance_np(query, key, radius, max_neighbors, transpose=True):
    index = []
    distance = []
    if transpose:
        query = query.transpose([0, 2, 1])
        key = key.transpose([0, 2, 1])
    n1 = query.shape[1]
    # n2 = key.shape[1]

    for query_per_batch, key_per_batch in zip(query, key):
        index_per_batch = np.full([n1, max_neighbors], -1, dtype=np.int64)
        distance_per_batch = np.full([n1, max_neighbors], -1.0, dtype=np.float32)
        for i in range(n1):
            cur_query = query_per_batch[i]
            dist2cur = key_per_batch - cur_query[None, :]
            dist2cur = np.square(dist2cur).sum(1)
            neighbor_index = np.nonzero(dist2cur < (radius ** 2))[0]
            assert neighbor_index.size > 0
            if neighbor_index.size < max_neighbors:
                index_per_batch[i, :neighbor_index.size] = neighbor_index
                index_per_batch[i, neighbor_index.size:] = neighbor_index[0]
                distance_per_batch[i, :neighbor_index.size] = dist2cur[neighbor_index]
            else:
                index_per_batch[i, :] = neighbor_index[:max_neighbors]
                distance_per_batch[i, :] = dist2cur[neighbor_index[:max_neighbors]]
        index.append(index_per_batch)
        distance.append(distance_per_batch)
    return np.asarray(index), np.asarray(distance)


@pytest.mark.parametrize('b, n1, n2, r, k, transpose, profile', test_data)
def test_ball_query_distance(b, n1, n2, r, k, transpose, profile):
    np.random.seed(0)
    if transpose:
        key = np.random.randn(b, 3, n2)
        query = np.array([p[:, np.random.choice(n2, n1, replace=False)] for p in key])
    else:
        key = np.random.randn(b, n2, 3)
        query = np.array([p[np.random.choice(n2, n1, replace=False)] for p in key])
    # key = key.astype(np.float32)
    # query = query.astype(np.float32)
    index_np, distance_np = ball_query_distance_np(query, key, r, k, transpose=transpose)
    # index_np = index_np.astype(np.int64)
    # distance_np = distance_np.astype(np.float32)

    query_tensor = torch.tensor(query).cuda()
    key_tensor = torch.tensor(key).cuda()
    index_tensor, distance_tensor = ball_query_distance(query_tensor, key_tensor, r, k, transpose=transpose)
    np.testing.assert_equal(index_np, index_tensor.cpu().numpy())
    np.testing.assert_allclose(distance_np, distance_tensor.cpu().numpy())

    if profile:
        query_tensor = query_tensor.float()
        key_tensor = key_tensor.float()
        with torch.autograd.profiler.profile(use_cuda=torch.cuda.is_available()) as prof:
            ball_query_distance(query_tensor, key_tensor, r, k, transpose=transpose)
        print(prof)
