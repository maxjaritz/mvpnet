import torch
from . import ball_query_cuda
from . import ball_query_distance_cuda


class BallQueryFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, radius, max_neighbors):
        index = ball_query_cuda.ball_query(query, key, radius, max_neighbors)
        return index

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None,) * len(grad_outputs)


def ball_query(query, key, radius, max_neighbors, transpose=True):
    if transpose:
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
    query = query.contiguous()
    key = key.contiguous()
    index = BallQueryFunction.apply(query, key, radius, max_neighbors)
    return index


class BallQueryDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, radius, max_neighbors):
        index, distance = ball_query_distance_cuda.ball_query_distance(query, key, radius, max_neighbors)
        return index, distance

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None,) * len(grad_outputs)


def ball_query_distance(query, key, radius, max_neighbors, transpose=True):
    if transpose:
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
    query = query.contiguous()
    key = key.contiguous()
    index, distance = BallQueryDistanceFunction.apply(query, key, radius, max_neighbors)
    return index, distance
