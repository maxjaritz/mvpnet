"""PointNet2(single-scale grouping)

References:
    @article{qi2017pointnetplusplus,
      title={PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space},
      author={Qi, Charles R and Yi, Li and Su, Hao and Guibas, Leonidas J},
      journal={arXiv preprint arXiv:1706.02413},
      year={2017}
    }

"""

import numpy as np
import torch
import torch.nn as nn

from common.nn import SharedMLPDO
from common.nn.init import xavier_uniform
from mvpnet.models.pn2.modules import SetAbstraction, FeaturePropagation


class PN2SSG(nn.Module):
    def __init__(self,
                 in_channels,
                 num_classes,
                 sa_channels=((32, 32, 64), (64, 64, 128), (128, 128, 256), (256, 256, 512)),
                 num_centroids=(2048, 512, 128, 32),
                 radius=(0.1, 0.2, 0.4, 0.8),
                 max_neighbors=(32, 32, 32, 32),
                 fp_channels=((256, 256), (256, 256), (256, 128), (128, 128, 128)),
                 fp_neighbors=(3, 3, 3, 3),
                 seg_channels=(128,),
                 dropout_prob=0.5,
                 use_xyz=True):
        super(PN2SSG, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.use_xyz = use_xyz

        # sanity check
        num_sa_layers = len(sa_channels)
        num_fp_layers = len(fp_channels)
        assert len(num_centroids) == num_sa_layers
        assert len(radius) == num_sa_layers
        assert len(max_neighbors) == num_sa_layers
        assert num_sa_layers == num_fp_layers
        assert len(fp_neighbors) == num_fp_layers

        # Set Abstraction Layers
        c_in = in_channels
        self.sa_modules = nn.ModuleList()
        for ind in range(num_sa_layers):
            sa_module = SetAbstraction(in_channels=c_in,
                                       mlp_channels=sa_channels[ind],
                                       num_centroids=num_centroids[ind],
                                       radius=radius[ind],
                                       max_neighbors=max_neighbors[ind],
                                       use_xyz=use_xyz)
            self.sa_modules.append(sa_module)
            c_in = sa_channels[ind][-1]

        # Get channels for all the intermediate features
        # Ignore the input feature
        # feature_channels = [self.in_channels]
        feature_channels = [0]
        feature_channels.extend([x[-1] for x in sa_channels])

        # Feature Propagation Layers
        c_in = feature_channels[-1]
        self.fp_modules = nn.ModuleList()
        for ind in range(num_fp_layers):
            fp_module = FeaturePropagation(in_channels=c_in,
                                           in_channels_prev=feature_channels[-2 - ind],
                                           mlp_channels=fp_channels[ind],
                                           num_neighbors=fp_neighbors[ind])
            self.fp_modules.append(fp_module)
            c_in = fp_channels[ind][-1]

        # MLP
        self.mlp_seg = SharedMLPDO(fp_channels[-1][-1], seg_channels, ndim=1, bn=True, p=dropout_prob)
        self.seg_logit = nn.Conv1d(seg_channels[-1], num_classes, 1, bias=True)

        # Initialize
        self.reset_parameters()

    def forward(self, data_batch):
        xyz = data_batch['points']
        feature = data_batch.get('feature', None)
        preds = dict()

        xyz_list = [xyz]
        # sa_feature_list = [feature]
        sa_feature_list = [None]

        # Set Abstraction Layers
        for sa_ind, sa_module in enumerate(self.sa_modules):
            xyz, feature = sa_module(xyz, feature)
            xyz_list.append(xyz)
            sa_feature_list.append(feature)

        # Feature Propagation Layers
        fp_feature_list = []
        for fp_ind, fp_module in enumerate(self.fp_modules):
            fp_feature = fp_module(
                xyz_list[-2 - fp_ind],
                xyz_list[-1 - fp_ind],
                sa_feature_list[-2 - fp_ind],
                fp_feature_list[-1] if len(fp_feature_list) > 0 else sa_feature_list[-1],
            )
            fp_feature_list.append(fp_feature)

        # MLP
        seg_feature = self.mlp_seg(fp_feature_list[-1])
        seg_logit = self.seg_logit(seg_feature)

        preds['seg_logit'] = seg_logit
        return preds

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                xavier_uniform(m)

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
        metric_fn = lambda: [SegAccuracy(), SegIoU(self.num_classes)]
        return metric_fn(), metric_fn()


def test(b=2, c=0, n=8192):
    data_batch = dict()
    data_batch['points'] = torch.randn(b, 3, n)
    if c > 0:
        data_batch['feature'] = torch.randn(b, c, n)
    data_batch = {k: v.cuda() for k, v in data_batch.items()}

    net = PN2SSG(c, 20)
    net = net.cuda()
    print(net)
    preds = net(data_batch)
    for k, v in preds.items():
        print(k, v.shape)
