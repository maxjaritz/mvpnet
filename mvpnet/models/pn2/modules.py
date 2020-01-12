import torch
import torch.nn as nn

from common.nn import SharedMLP
from common.nn.functional import batch_index_select
from mvpnet.ops.fps import farthest_point_sample
from mvpnet.ops.group_points import group_points
from mvpnet.ops.ball_query import ball_query
from mvpnet.ops.knn_distance import knn_distance
from mvpnet.ops.interpolate import feature_interpolate


class QueryGrouper(nn.Module):
    def __init__(self, radius, max_neighbors):
        super(QueryGrouper, self).__init__()
        assert radius > 0.0 and max_neighbors > 0
        self.radius = radius
        self.max_neighbors = max_neighbors

    def forward(self, new_xyz, xyz, feature, use_xyz):
        with torch.no_grad():
            index = ball_query(new_xyz, xyz, self.radius, self.max_neighbors)

        # (batch_size, 3, num_centroids, num_neighbors)
        group_xyz = group_points(xyz, index)
        # translation normalization
        group_xyz -= new_xyz.unsqueeze(-1)

        if feature is not None:
            # (batch_size, channels, num_centroids, num_neighbors)
            group_feature = group_points(feature, index)
            if use_xyz:
                group_feature = torch.cat([group_feature, group_xyz], dim=1)
        else:
            group_feature = group_xyz

        return group_feature, group_xyz

    def extra_repr(self):
        attributes = ['radius', 'max_neighbors']
        return ', '.join(['{:s}={}'.format(name, getattr(self, name)) for name in attributes])


class SetAbstraction(nn.Module):
    """PointNet++ set abstraction module"""

    def __init__(self,
                 in_channels,
                 mlp_channels,
                 num_centroids,
                 radius,
                 max_neighbors,
                 use_xyz):
        super(SetAbstraction, self).__init__()

        self.in_channels = in_channels
        self.out_channels = mlp_channels[-1]
        self.num_centroids = num_centroids
        self.radius = radius
        self.max_neighbors = max_neighbors
        self.use_xyz = use_xyz

        if self.use_xyz or self.in_channels == 0:
            self.in_channels += 3

        self.mlp = SharedMLP(self.in_channels, mlp_channels, ndim=2, bn=True)

        if num_centroids == 0:
            # use the origin as the centroid
            self.grouper = None
        else:
            self.grouper = QueryGrouper(radius, max_neighbors)

    def forward(self, xyz, feature=None):
        """

        Args:
            xyz (torch.Tensor): (batch_size, 3, num_points)
                xyz coordinates of feature
            feature (torch.Tensor, optional): (batch_size, in_channels, num_points)

        Returns:
            new_xyz (torch.Tensor): (batch_size, 3, num_centroids)
            new_feature (torch.Tensor): (batch_size, out_channels, num_centroids)

        """
        batch_size = xyz.size(0)
        if self.num_centroids == 0:  # use the origin as the centroid
            new_xyz = xyz.new_zeros([batch_size, 3, 1])
            assert self.grouper is None
            assert feature is not None
            group_feature = feature.unsqueeze(2)  # (batch_size, in_channels, 1, num_points)
            if self.use_xyz:
                group_xyz = xyz.unsqueeze(2)  # (batch_size, 3, 1, num_points)
                group_feature = torch.cat([group_feature, group_xyz], dim=1)
        else:
            if self.num_centroids == -1:  # no sampling
                new_xyz = xyz
            else:  # sample new points
                with torch.no_grad():
                    index = farthest_point_sample(xyz, self.num_centroids)  # (batch_size, num_centroids)
                new_xyz = batch_index_select(xyz, index, dim=2)  # (batch_size, 3, num_centroids)
            # group_feature: (batch_size, in_channels, num_centroids, num_neighbors)
            group_feature, group_xyz = self.grouper(new_xyz, xyz, feature, use_xyz=self.use_xyz)

        # Apply PointNet on local regions
        new_feature = self.mlp(group_feature)
        new_feature, _ = torch.max(new_feature, dim=3)
        return new_xyz, new_feature

    def extra_repr(self):
        attributes = ['num_centroids', 'radius', 'max_neighbors', 'use_xyz']
        return ', '.join(['{:s}={}'.format(name, getattr(self, name)) for name in attributes])


class FeatureInterpolator(nn.Module):
    def __init__(self, num_neighbors, eps=1e-10):
        super(FeatureInterpolator, self).__init__()
        self.num_neighbors = num_neighbors
        self._eps = eps

    def forward(self, query_xyz, key_xyz, query_feature, key_feature):
        """Interpolate features from key to query
        
        Args:
            query_xyz: (B, 3, N1)
            key_xyz: (B, 3, N2)
            query_feature: (B, C1, N1)
            key_feature: (B, C2, N2)

        Returns:
            new_feature: (B, C1+C2, N1), propagated feature

        """
        with torch.no_grad():
            # index: (B, N1, K), distance: (B, N1, K)
            index, distance = knn_distance(query_xyz, key_xyz, self.num_neighbors)
            inv_distance = 1.0 / torch.clamp(distance, min=self._eps)
            norm = torch.sum(inv_distance, dim=2, keepdim=True)
            weight = inv_distance / norm

        interpolated_feature = feature_interpolate(key_feature, index, weight)

        if query_feature is not None:
            new_feature = torch.cat([interpolated_feature, query_feature], dim=1)
        else:
            new_feature = interpolated_feature

        return new_feature

    def extra_repr(self):
        attributes = ['num_neighbors']
        return ', '.join(['{:s}={}'.format(name, getattr(self, name)) for name in attributes])


class FeaturePropagation(nn.Module):
    """PointNet feature propagation module"""

    def __init__(self,
                 in_channels,
                 in_channels_prev,
                 mlp_channels,
                 num_neighbors):
        super(FeaturePropagation, self).__init__()

        self.in_channels = in_channels + in_channels_prev
        self.out_channels = mlp_channels[-1]

        self.mlp = SharedMLP(self.in_channels, mlp_channels, ndim=1, bn=True)
        if num_neighbors == 0:
            # expand global features
            self.interpolator = None
        elif num_neighbors == 3:
            self.interpolator = FeatureInterpolator(num_neighbors)
        else:
            raise ValueError('Expected value 3, but {} given.'.format(num_neighbors))

    def forward(self, dense_xyz, sparse_xyz, dense_feature, sparse_feature):
        if self.interpolator is None:
            assert sparse_xyz.size(2) == 1 and sparse_feature.size(2) == 1
            sparse_feature_expand = sparse_feature.expand(-1, -1, dense_xyz.size(2))
            new_feature = torch.cat([sparse_feature_expand, dense_feature], dim=1)
        else:
            new_feature = self.interpolator(dense_xyz, sparse_xyz, dense_feature, sparse_feature)
        new_feature = self.mlp(new_feature)
        return new_feature
