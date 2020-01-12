"""Segmentation experiments configuration"""

from common.config.base import CN, _C

# public alias
cfg = _C
_C.TASK = 'sem_seg_2d'
_C.VAL.METRIC = 'seg_iou'

# ----------------------------------------------------------------------------- #
# Dataset
# ----------------------------------------------------------------------------- #
_C.DATASET.ROOT_DIR = ''
_C.DATASET.TRAIN = ''
_C.DATASET.VAL = ''

_C.DATASET.ScanNet2D = CN()
_C.DATASET.ScanNet2D.resize = ()
_C.DATASET.ScanNet2D.normalizer = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
_C.DATASET.ScanNet2D.augmentation = CN()
_C.DATASET.ScanNet2D.augmentation.flip = 0.0
_C.DATASET.ScanNet2D.augmentation.color_jitter = ()

# ---------------------------------------------------------------------------- #
# Specific validation options
# ---------------------------------------------------------------------------- #
_C.VAL.REPEATS = 1

# ---------------------------------------------------------------------------- #
# UNetResNet34 options
# ---------------------------------------------------------------------------- #
_C.MODEL.UNetResNet34 = CN()
_C.MODEL.UNetResNet34.num_classes = 20
_C.MODEL.UNetResNet34.p = 0.0

