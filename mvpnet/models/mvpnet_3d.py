import torch
from torch import nn
import numpy as np

from common.nn import SharedMLP
from mvpnet.ops.group_points import group_points


class FeatureAggregation(nn.Module):
    """Feature Aggregation inspired by ContFuse"""

    def __init__(self,
                 in_channels,
                 mlp_channels=(64, 64, 64),
                 reduction='sum',
                 use_relation=True,
                 ):
        super(FeatureAggregation, self).__init__()

        self.in_channels = in_channels
        self.use_relation = use_relation

        if mlp_channels:
            self.out_channels = mlp_channels[-1]
            self.mlp = SharedMLP(in_channels + (4 if use_relation else 0), mlp_channels, ndim=2, bn=True)
        else:
            self.out_channels = in_channels
            self.mlp = None

        if reduction == 'sum':
            self.reduction = torch.sum
        elif reduction == 'max':
            self.reduction = lambda x, dim: torch.max(x, dim)[0]

        self.reset_parameters()

    def forward(self, src_xyz, tgt_xyz, feature):
        """

        Args:
            src_xyz (torch.Tensor): (batch_size, 3, num_points, k)
            tgt_xyz (torch.Tensor): (batch_size, 3, num_points)
            feature (torch.Tensor): (batch_size, in_channels, num_points, k)

        Returns:
            torch.Tensor: (batch_size, out_channels, num_points)

        """
        if self.mlp is not None:
            if self.use_relation:
                diff_xyz = src_xyz - tgt_xyz.unsqueeze(-1)  # (b, 3, np, k)
                distance = torch.sum(diff_xyz ** 2, dim=1, keepdim=True)  # (b, 1, np, k)
                relation_feature = torch.cat([diff_xyz, distance], dim=1)
                x = torch.cat([feature, relation_feature], 1)
            else:
                x = feature
            x = self.mlp(x)
            x = self.reduction(x, 3)
        else:
            x = self.reduction(feature, 3)
        return x

    def reset_parameters(self):
        from common.nn.init import xavier_uniform
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                xavier_uniform(m)


class MVPNet3D(nn.Module):
    def __init__(self,
                 net_2d,
                 net_2d_ckpt_path,
                 net_3d,
                 **feat_aggr_kwargs,
                 ):
        super(MVPNet3D, self).__init__()
        self.net_2d = net_2d
        if net_2d_ckpt_path:
            checkpoint = torch.load(net_2d_ckpt_path, map_location=torch.device("cpu"))
            self.net_2d.load_state_dict(checkpoint['model'])
            import logging
            logger = logging.getLogger(__name__)
            logger.info("2D network load weights from {}.".format(net_2d_ckpt_path))
        self.feat_aggreg = FeatureAggregation(**feat_aggr_kwargs)
        self.net_3d = net_3d

    def forward(self, data_batch):
        # (batch_size, num_views, 3, h, w)
        images = data_batch['images']
        b, nv, _, h, w = images.size()
        # collapse first 2 dimensions together
        images = images.reshape([-1] + list(images.shape[2:]))

        # 2D network
        preds_2d = self.net_2d({'image': images})
        feature_2d = preds_2d['feature']  # (b * nv, c, h, w)

        # unproject features
        knn_indices = data_batch['knn_indices']  # (b, np, k)
        feature_2d = feature_2d.reshape(b, nv, -1, h, w).transpose(1, 2).contiguous()  # (b, c, nv, h, w)
        feature_2d = feature_2d.reshape(b, -1, nv * h * w)
        feature_2d = group_points(feature_2d, knn_indices)  # (b, c, np, k)

        # unproject depth maps
        with torch.no_grad():
            image_xyz = data_batch['image_xyz']  # (b, nv, h, w, 3)
            image_xyz = image_xyz.permute(0, 4, 1, 2, 3).reshape(b, 3, nv * h * w)
            image_xyz = group_points(image_xyz, knn_indices)  # (b, 3, np, k)

        # 2D-3D aggregation
        points = data_batch['points']
        feature_2d3d = self.feat_aggreg(image_xyz, points, feature_2d)

        # 3D network
        preds_3d = self.net_3d({'points': points, 'feature': feature_2d3d})
        preds = preds_3d
        return preds

    def get_loss(self, cfg):
        from mvpnet.models.loss import SegLoss
        if cfg.TRAIN.LABEL_WEIGHTS_PATH:
            weights = np.loadtxt(cfg.TRAIN.LABEL_WEIGHTS_PATH, dtype=np.float32)
            weights = torch.from_numpy(weights).cuda()
        else:
            weights = None
        return SegLoss(weight=weights)

    def get_metric(self, cfg):
        from mvpnet.models.metric import SegAccuracy, SegIoU
        metric_fn = lambda: [SegAccuracy(), SegIoU(self.net_3d.num_classes)]
        return metric_fn(), metric_fn()
