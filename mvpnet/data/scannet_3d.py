import os.path as osp
from collections import OrderedDict
import pickle
import logging

import numpy as np
from torch.utils.data import Dataset

_CUR_DIR = osp.dirname(__file__)
_META_DIR = osp.abspath(osp.join(_CUR_DIR, 'meta_files'))


def load_class_mapping(filename):
    id_to_class = OrderedDict()
    with open(filename, 'r') as f:
        for line in f.readlines():
            class_id, class_name = line.rstrip().split('\t')
            id_to_class[int(class_id)] = class_name
    return id_to_class


class ScanNet3D(Dataset):
    """ScanNetV2 3D dataset. Base class to provide raw data.
    ScanNetV2 annotates each point with nyu40 classes (indexed from 1). 0 for unannotated.
    """
    nyu40_classes_path = osp.join(_META_DIR, 'labelids_all.txt')
    scannet_classes_path = osp.join(_META_DIR, 'labelids.txt')
    split_dir = _META_DIR
    split_map = {
        'train': 'scannetv2_train.txt',
        'val': 'scannetv2_val.txt',
        'test': 'scannetv2_test.txt',
    }
    ignore_value = -100

    def __init__(self, root_dir, split):
        self.root_dir = root_dir
        self.split = split

        # load split
        with open(osp.join(self.split_dir, self.split_map[split]), 'r') as f:
            self.scan_ids = [line.rstrip() for line in f.readlines()]

        self.class_names = None
        self.label_mapping = None
        # nyu40
        self.nyu40_mapping = load_class_mapping(self.nyu40_classes_path)
        assert len(self.nyu40_mapping) == 40
        # scannet
        self.scannet_mapping = load_class_mapping(self.scannet_classes_path)
        assert len(self.scannet_mapping) == 20
        # mapping
        self.raw_to_scannet = np.full(shape=41, fill_value=self.ignore_value, dtype=np.int64)
        self.raw_to_scannet[list(self.scannet_mapping.keys())] = np.arange(len(self.scannet_mapping))
        self.scannet_to_raw = np.array(list(self.scannet_mapping.keys()) + [0], dtype=np.int64)
        self.set_mapping(None)

        # We load the data into the memory since the size of raw 3d data is only a few GB.
        with open(osp.join(self.root_dir, 'scannetv2_{}.pkl'.format(self.split)), 'rb') as f:
            self.data = pickle.load(f)
        assert len(self.data) == len(self.scan_ids)

    def set_mapping(self, style):
        if style is None:
            self.class_names = ('unannotated',) + tuple(self.nyu40_mapping.values())
            self.label_mapping = None
        elif style == 'scannet':
            self.class_names = tuple(self.scannet_mapping.values())
            self.label_mapping = self.raw_to_scannet
        elif style == 'nyu40':
            self.class_names = tuple(self.nyu40_mapping.values())
            self.label_mapping = np.array([self.ignore_value] + [i for i in range(40)])
        else:
            raise ValueError('Unknown style: {}'.format(style))

    def __getitem__(self, index):
        data_dict = self.data[index]
        scan_id = data_dict['scan_id']
        assert scan_id == self.scan_ids[index], 'Mismatch scan_id: {} vs {}.'.format(scan_id, self.scan_ids[index])
        # Note that astype will copy data.
        points = data_dict['points'].astype(np.float32)
        colors = data_dict['colors'].astype(np.float32) / 255.

        out_dict = {
            'scan_id': scan_id,
            'points': points,
            'colors': colors,
        }

        # If labels are provided
        seg_label = data_dict.get('seg_label', None)
        if seg_label is not None:
            if self.label_mapping is not None:
                seg_label = self.label_mapping[seg_label]
            out_dict['seg_label'] = seg_label.astype(np.int64)

        return out_dict

    def __len__(self):
        return len(self.scan_ids)

    def __str__(self):
        return '{:s}: {} classes with {} scenes'.format(
            self.__class__.__name__, len(self.class_names), len(self))


class ScanNet3DChunks(ScanNet3D):
    """ScanNetV2 3D chunks
    Notes: The point cloud in ScanNet is not aligned, and thus it is tricky to detect floors by z coord.
    """

    def __init__(self,
                 root_dir,
                 split,
                 use_color=False,
                 transform=None,
                 chunk_size=(1.5, 1.5),
                 chunk_thresh=0.3,
                 chunk_margin=(0.2, 0.2),
                 ):
        super(ScanNet3DChunks, self).__init__(root_dir, split)
        self.set_mapping('scannet')

        self.use_color = use_color
        self.transform = transform

        # The height / z-axis is ignored in fact.
        self.chunk_size = np.array(chunk_size)
        self.chunk_thresh = chunk_thresh
        self.chunk_margin = np.array(chunk_margin)

        logger = logging.getLogger(__name__)
        logger.info(str(self))

    def __getitem__(self, index):
        data_dict = self.data[index]
        scan_id = data_dict['scan_id']
        assert scan_id == self.scan_ids[index], 'Mismatch scan_id: {} vs {}.'.format(scan_id, self.scan_ids[index])
        # Note that astype will copy data.
        points = data_dict['points'].astype(np.float32)
        seg_label = data_dict['seg_label'].astype(np.int64)
        if self.label_mapping is not None:
            seg_label = self.label_mapping[seg_label]

        # ---------------------------------------------------------------------------- #
        # Choose a random chunk
        # ---------------------------------------------------------------------------- #
        # Try several times. If it fails then returns the whole scene
        flag = False
        for _ in range(10):
            # choose a random center (only xy)
            center = points[np.random.randint(points.shape[0])][:2]
            # determine region around center
            chunk_min = center - 0.5 * self.chunk_size
            chunk_max = center + 0.5 * self.chunk_size

            # Magic numbers are referred to the original implementation.
            # select points within the block with a margin
            xy = points[:, :2]
            chunk_mask = np.all(np.logical_and(xy >= (chunk_min - self.chunk_margin),
                                               xy <= (chunk_max + self.chunk_margin)),
                                axis=1)
            # actual points and seg labels as determined above
            chunk_points = points[chunk_mask, :]
            chunk_seg_label = seg_label[chunk_mask]
            # continue if no points are found
            if len(chunk_seg_label) == 0:
                continue
            if np.mean(chunk_seg_label >= 0) >= self.chunk_thresh:
                flag = True
                break

        if not flag:
            chunk_points = points
            chunk_seg_label = seg_label
            chunk_mask = np.ones_like(seg_label, dtype=bool)
            print('No valid chunk found in scene {}. '
                  'Return all points({:d}).'.format(scan_id, points.shape[0]))

        out_dict = {
            'points': chunk_points,
            'seg_label': chunk_seg_label,
        }
        if self.use_color:
            out_dict['feature'] = data_dict['colors'][chunk_mask].astype(np.float32) / 255.

        if self.transform is not None:
            out_dict = self.transform(**out_dict)
        return out_dict


class ScanNet3DScene(ScanNet3D):
    def __init__(self,
                 root_dir,
                 split,
                 use_color=False,
                 transform=None):
        super(ScanNet3DScene, self).__init__(root_dir, split)
        self.set_mapping('scannet')
        self.use_color = use_color
        self.transform = transform

        logger = logging.getLogger(__name__)
        logger.info(str(self))

    def __getitem__(self, index):
        data_dict = super(ScanNet3DScene, self).__getitem__(index)
        points = data_dict['points']
        seg_label = data_dict['seg_label']
        data_dict.pop('scan_id')

        out_dict = {
            'points': points,
            'seg_label': seg_label,
        }
        if self.use_color:
            out_dict['feature'] = data_dict['colors']

        if self.transform is not None:
            out_dict = self.transform(**out_dict)
        return out_dict


def centralize_chunk(points):
    """In-place centralize a chunk"""
    assert points.ndim == 2 and points.shape[1] >= 3
    chunk_min = points[:, 0:3].min(0)
    chunk_max = points[:, 0:3].max(0)
    points[:, 0:2] -= 0.5 * (chunk_min + chunk_max)[0:2]
    points[:, 2] -= chunk_min[2]
    return points


def centralize_scene(points):
    """In-place centralize a whole scene"""
    assert points.ndim == 2 and points.shape[1] >= 3
    points[:, 0:2] -= points[:, 0:2].mean(0)
    points[:, 2] -= points[:, 2].min(0)
    return points


def test_ScanNet3D():
    from mvpnet.utils.o3d_util import visualize_point_cloud
    from mvpnet.utils.visualize import visualize_labels
    data_dir = osp.join(_CUR_DIR, '../../data/ScanNet/cache_3d')
    split = 'test'
    dataset = ScanNet3D(data_dir, split)
    dataset.set_mapping('scannet')
    # dataset.set_mapping('nyu40')

    for i in range(len(dataset)):
        data = dataset[i]
        points = data['points']
        seg_label = data.get('seg_label', None)
        colors = data['colors']
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                print(k, v.shape, v.dtype)
            else:
                print(k, v)
        visualize_point_cloud(points, colors, show_frame=True)
        if split != 'test':
            # visualize_labels(points, seg_label, colors, style='nyu40_raw')
            visualize_labels(points, seg_label, colors, style='scannet')


def test_ScanNet3DChunks():
    from mvpnet.data.transforms import Compose, CropPad, RandomRotateZ
    from mvpnet.utils.visualize import visualize_labels
    data_dir = osp.join(_CUR_DIR, '../../data/ScanNet/cache_3d')

    transform = None
    transform = Compose([CropPad(8192), RandomRotateZ()])
    dataset = ScanNet3DChunks(data_dir, 'val', use_color=True, transform=transform)
    dataset.set_mapping('scannet')
    for i in range(len(dataset)):
        data = dataset[i]
        points = data['points']
        seg_label = data['seg_label']
        colors = data.get('feature', None)
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                print(k, v.shape, v.dtype)
            else:
                print(k, v)
        visualize_labels(points, seg_label, colors, style='scannet')


def test_ScanNet3DScene():
    from mvpnet.data.transforms import Compose, CropPad, RandomRotateZ
    from mvpnet.utils.visualize import visualize_labels
    data_dir = osp.join(_CUR_DIR, '../../data/ScanNet/cache_3d')

    # transform = None
    transform = Compose([CropPad(32768), RandomRotateZ()])
    dataset = ScanNet3DScene(data_dir, 'val', use_color=True, transform=transform)
    dataset.set_mapping('scannet')
    for i in range(len(dataset)):
        data = dataset[i]
        points = data['points']
        seg_label = data['seg_label']
        colors = data.get('feature', None)
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                print(k, v.shape, v.dtype)
            else:
                print(k, v)
        visualize_labels(points, seg_label, colors, style='scannet')
