import torch
from torch import nn

from mvpnet.ops.group_points import group_points


class MVPNet2D(nn.Module):
    def __init__(self, net_2d):
        super(MVPNet2D, self).__init__()
        self.net_2d = net_2d

    def forward(self, data_batch):
        # (batch_size, num_views, 3, h, w)
        images = data_batch['images']
        b, nv, _, h, w = images.size()
        # collapse first 2 dimensions together
        images = images.reshape([-1] + list(images.shape[2:]))

        # 2D network
        preds_2d = self.net_2d({'image': images})
        seg_logit_2d = preds_2d['seg_logit']  # (b * nv, nc, h, w)
        # feature_2d = preds_2d['feature']  # (b * nv, c, h, w)

        # unproject features
        knn_indices = data_batch['knn_indices']  # (b, np, k)
        seg_logit = seg_logit_2d.reshape(b, nv, -1, h, w).transpose(1, 2).contiguous()  # (b, nc, nv, h, w)
        seg_logit = seg_logit.reshape(b, -1, nv * h * w)
        seg_logit = group_points(seg_logit, knn_indices)  # (b, nc, np, k)
        seg_logit = seg_logit.mean(-1)  # (b, nc, np)

        preds = {
            'seg_logit': seg_logit,
        }
        return preds
