"""Segmentation experiments configuration"""

from common.config.base import CN, _C

# public alias
cfg = _C
_C.TASK = 'mvpnet_3d'
_C.VAL.METRIC = 'seg_iou'

# ----------------------------------------------------------------------------- #
# Dataset
# ----------------------------------------------------------------------------- #
_C.DATASET.TRAIN = ''
_C.DATASET.VAL = ''

# Chunk-based
_C.DATASET.ScanNet2D3DChunks = CN()
_C.DATASET.ScanNet2D3DChunks.cache_dir = ''
_C.DATASET.ScanNet2D3DChunks.image_dir = ''
_C.DATASET.ScanNet2D3DChunks.chunk_size = (1.5, 1.5)
_C.DATASET.ScanNet2D3DChunks.chunk_thresh = 0.3
_C.DATASET.ScanNet2D3DChunks.chunk_margin = (0.2, 0.2)
_C.DATASET.ScanNet2D3DChunks.nb_pts = 8192
_C.DATASET.ScanNet2D3DChunks.num_rgbd_frames = 3
_C.DATASET.ScanNet2D3DChunks.resize = (160, 120)
_C.DATASET.ScanNet2D3DChunks.image_normalizer = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
_C.DATASET.ScanNet2D3DChunks.k = 3
_C.DATASET.ScanNet2D3DChunks.augmentation = CN()
_C.DATASET.ScanNet2D3DChunks.augmentation.z_rot = ()  # degree instead of rad
_C.DATASET.ScanNet2D3DChunks.augmentation.flip = 0.0
_C.DATASET.ScanNet2D3DChunks.augmentation.color_jitter = ()

# ---------------------------------------------------------------------------- #
# Specific validation options
# ---------------------------------------------------------------------------- #
_C.VAL.REPEATS = 1

# ---------------------------------------------------------------------------- #
# Model 3D
# ---------------------------------------------------------------------------- #
_C.MODEL_3D = CN()
_C.MODEL_3D.TYPE = ''
_C.MODEL_3D.TYPE = ''
# ---------------------------------------------------------------------------- #
# PN2SSG options
# ---------------------------------------------------------------------------- #
_C.MODEL_3D.PN2SSG = CN()
_C.MODEL_3D.PN2SSG.in_channels = 64  # match feature aggregation
_C.MODEL_3D.PN2SSG.num_classes = 20
_C.MODEL_3D.PN2SSG.sa_channels = ((32, 32, 64), (64, 64, 128), (128, 128, 256), (256, 256, 512))
_C.MODEL_3D.PN2SSG.num_centroids = (2048, 512, 128, 32)
_C.MODEL_3D.PN2SSG.radius = (0.1, 0.2, 0.4, 0.8)
_C.MODEL_3D.PN2SSG.max_neighbors = (32, 32, 32, 32)
_C.MODEL_3D.PN2SSG.fp_channels = ((256, 256), (256, 256), (256, 128), (128, 128, 128))
_C.MODEL_3D.PN2SSG.fp_neighbors = (3, 3, 3, 3)
_C.MODEL_3D.PN2SSG.seg_channels = (128,)
_C.MODEL_3D.PN2SSG.dropout_prob = 0.5
_C.MODEL_3D.PN2SSG.use_xyz = True

# ---------------------------------------------------------------------------- #
# Model 2D
# ---------------------------------------------------------------------------- #
_C.MODEL_2D = CN()
_C.MODEL_2D.TYPE = ''
_C.MODEL_2D.CKPT_PATH = ''
# ---------------------------------------------------------------------------- #
# UNetResNet34 options
# ---------------------------------------------------------------------------- #
_C.MODEL_2D.UNetResNet34 = CN()
_C.MODEL_2D.UNetResNet34.num_classes = 20
_C.MODEL_2D.UNetResNet34.p = 0.0

# ---------------------------------------------------------------------------- #
# Feature Aggregation
# ---------------------------------------------------------------------------- #
_C.FEAT_AGGR = CN()
_C.FEAT_AGGR.in_channels = 64  # match 2D network
_C.FEAT_AGGR.mlp_channels = (64, 64, 64)
_C.FEAT_AGGR.reduction = 'sum'
_C.FEAT_AGGR.use_relation = True
