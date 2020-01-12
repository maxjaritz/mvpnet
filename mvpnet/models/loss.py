import torch.nn as nn
import torch.nn.functional as F


class SegLoss(nn.Module):
    """Segmentation loss"""

    def __init__(self, weight=None, ignore_index=-100):
        super(SegLoss, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, preds, labels):
        loss_dict = dict()
        logits = preds["seg_logit"]
        labels = labels["seg_label"]
        seg_loss = F.cross_entropy(logits, labels,
                                   weight=self.weight,
                                   ignore_index=self.ignore_index)
        loss_dict['seg_loss'] = seg_loss
        return loss_dict
