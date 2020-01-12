"""Segmentation experiments configuration"""

from common.config.base import CN, _C

# public alias
cfg = _C
_C.TASK = 'sem_seg_3d'
_C.VAL.METRIC = 'seg_iou'

# ----------------------------------------------------------------------------- #
# Dataset
# ----------------------------------------------------------------------------- #
_C.DATASET.ROOT_DIR = ''
_C.DATASET.TRAIN = ''
_C.DATASET.VAL = ''

# Chunk-based
_C.DATASET.ScanNet3DChunks = CN()
_C.DATASET.ScanNet3DChunks.use_color = False
_C.DATASET.ScanNet3DChunks.chunk_size = (1.5, 1.5)
_C.DATASET.ScanNet3DChunks.chunk_thresh = 0.3
_C.DATASET.ScanNet3DChunks.chunk_margin = (0.2, 0.2)
# Scene-based
_C.DATASET.ScanNet3DScene = CN()
_C.DATASET.ScanNet3DScene.use_color = False

# ---------------------------------------------------------------------------- #
# Specific validation options
# ---------------------------------------------------------------------------- #
_C.VAL.REPEATS = 1

# ---------------------------------------------------------------------------- #
# PN2SSG options
# ---------------------------------------------------------------------------- #
_C.MODEL.PN2SSG = CN()
_C.MODEL.PN2SSG.in_channels = 0
_C.MODEL.PN2SSG.num_classes = 20
_C.MODEL.PN2SSG.sa_channels = ((32, 32, 64), (64, 64, 128), (128, 128, 256), (256, 256, 512))
_C.MODEL.PN2SSG.num_centroids = (2048, 512, 128, 32)
_C.MODEL.PN2SSG.radius = (0.1, 0.2, 0.4, 0.8)
_C.MODEL.PN2SSG.max_neighbors = (32, 32, 32, 32)
_C.MODEL.PN2SSG.fp_channels = ((256, 256), (256, 256), (256, 128), (128, 128, 128))
_C.MODEL.PN2SSG.fp_neighbors = (3, 3, 3, 3)
_C.MODEL.PN2SSG.seg_channels = (128,)
_C.MODEL.PN2SSG.dropout_prob = 0.5
_C.MODEL.PN2SSG.use_xyz = True
