"""UNet based on ResNet34"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet34


class UNetResNet34(nn.Module):
    def __init__(self, num_classes, p=0.0, pretrained=True):
        super(UNetResNet34, self).__init__()
        self.num_classes = num_classes

        # ----------------------------------------------------------------------------- #
        # Encoder
        # ----------------------------------------------------------------------------- #
        net = resnet34(pretrained)
        # Note that we do not downsample for conv1
        self.encoder0 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.encoder0.weight.data = net.conv1.weight.data
        # self.conv1 = net.conv1
        self.bn = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool
        self.encoder1 = net.layer1
        self.encoder2 = net.layer2
        self.encoder3 = net.layer3
        self.encoder4 = net.layer4
        # self.avgpool = net.avgpool

        # ----------------------------------------------------------------------------- #
        # Decoder
        # ----------------------------------------------------------------------------- #
        self.deconv4 = self.get_deconv(512, 256)
        self.decoder3 = self.get_conv(512, 256)
        self.deconv3 = self.get_deconv(256, 128)
        self.decoder2 = self.get_conv(256, 128)
        self.deconv2 = self.get_deconv(128, 64)
        self.decoder1 = self.get_conv(128, 64)
        self.deconv1 = self.get_deconv(64, 64)
        self.decoder0 = self.get_conv(128, 64)

        # logit
        self.logit = nn.Conv2d(64, num_classes, 1, bias=True)
        self.dropout = nn.Dropout(p=p) if p > 0.0 else None

    @staticmethod
    def get_deconv(c_in, c_out):
        deconv = nn.Sequential(
            nn.ConvTranspose2d(c_in, c_out, kernel_size=2, stride=2),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True)
        )
        return deconv

    @staticmethod
    def get_conv(c_in, c_out):
        conv = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )
        return conv

    def forward(self, data_dict):
        x = data_dict['image']
        h, w = x.shape[2], x.shape[3]
        # padding
        min_size = 16
        pad_h = int((h + min_size - 1) / min_size) * min_size - h
        pad_w = int((w + min_size - 1) / min_size) * min_size - w
        if pad_h > 0 or pad_w > 0:
            # Pad 0 here. Not sure whether has a large effect
            x = F.pad(x, [0, pad_w, 0, pad_h])
        # assert h % 16 == 0 and w % 16 == 0

        preds = dict()

        # ----------------------------------------------------------------------------- #
        # Encoder
        # ----------------------------------------------------------------------------- #
        encoder_features = []
        x = self.encoder0(x)
        x = self.bn(x)
        x = self.relu(x)
        encoder_features.append(x)
        x = self.maxpool(x)
        x = self.encoder1(x)
        encoder_features.append(x)
        x = self.encoder2(x)
        encoder_features.append(x)
        x = self.encoder3(x)
        # dropout
        if self.dropout is not None:
            x = self.dropout(x)
        encoder_features.append(x)
        x = self.encoder4(x)
        # dropout
        if self.dropout is not None:
            x = self.dropout(x)

        # ----------------------------------------------------------------------------- #
        # Decoder
        # ----------------------------------------------------------------------------- #
        x = self.deconv4(x)  # dim=512
        x = torch.cat([x, encoder_features[3]], dim=1)  # dim=512
        x = self.decoder3(x)  # dim=256
        x = self.deconv3(x)  # dim=128
        x = torch.cat([x, encoder_features[2]], dim=1)
        x = self.decoder2(x)
        x = self.deconv2(x)  # dim=64
        x = torch.cat([x, encoder_features[1]], dim=1)
        x = self.decoder1(x)
        x = self.deconv1(x)  # dim=64
        x = torch.cat([x, encoder_features[0]], dim=1)
        x = self.decoder0(x)

        # crop
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, 0:h, 0:w]

        seg_logit = self.logit(x)
        preds['seg_logit'] = seg_logit
        preds['feature'] = x
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
        metric_fn = lambda: [SegAccuracy(), SegIoU(self.num_classes)]
        return metric_fn(), metric_fn()

def test():
    b, c, h, w = 2, 20, 120, 160
    image = torch.randn(b, 3, h, w).cuda()
    net = UNetResNet34(c, pretrained=True)
    net.cuda()
    preds = net({'image': image})
    for k, v in preds.items():
        print(k, v.shape)
