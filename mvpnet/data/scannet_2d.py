import os
import os.path as osp
import glob
import csv
import json
import natsort
import logging
from tqdm import tqdm
from collections import OrderedDict

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms as T
from torchvision.transforms import functional as F

_CUR_DIR = osp.dirname(__file__)
_META_DIR = osp.abspath(osp.join(_CUR_DIR, 'meta_files'))


def read_label_mapping(filename, label_from='raw_category', label_to='nyu40id', as_int=False):
    """Read the mapping of labels from the given tsv"""
    assert osp.isfile(filename)
    mapping = dict()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            key = row[label_from]
            if as_int:
                key = int(key)
            mapping[key] = int(row[label_to])
    return mapping


def load_class_mapping(filename):
    id_to_class = OrderedDict()
    with open(filename, 'r') as f:
        for line in f.readlines():
            class_id, class_name = line.rstrip().split('\t')
            id_to_class[int(class_id)] = class_name
    return id_to_class


class ScanNet2D(Dataset):
    """ScanNetV2 2D dataset"""
    label_id_tsv_path = osp.join(_META_DIR, 'scannetv2-labels.combined.tsv')
    scannet_classes_path = osp.join(_META_DIR, 'labelids.txt')
    split_dir = _META_DIR
    split_map = {
        'train': 'scannetv2_train.txt',
        'val': 'scannetv2_val.txt',
        'test': 'scannetv2_test.txt',
    }
    # exclude some frames with problematic data (e.g. depth frames with zeros everywhere or unreadable labels)
    exclude_frames = {
        'scene0243_00': ['1175', '1176', '1177', '1178', '1179', '1180', '1181', '1182', '1183', '1184'],
        'scene0538_00': ['1925', '1928', '1929', '1931', '1932', '1933'],
        'scene0639_00': ['442', '443', '444'],
        'scene0299_01': ['1512'],
    }
    # scans_name = 'scans'
    scans_name = 'scans_resize_{}x{}'
    ignore_value = -100

    def __init__(self, root_dir, split,
                 to_tensor=False, subsample=None,
                 resize=None, normalizer=None,
                 flip=0.0, color_jitter=None,
                 ):
        self.root_dir = root_dir
        self.split = split
        # augmentation
        self.resize = resize
        self.normalizer = normalizer
        self.flip = flip
        self.color_jitter = T.ColorJitter(*color_jitter) if color_jitter else None
        # status
        self.to_tensor = to_tensor
        self.subsample = subsample

        # load split
        with open(osp.join(self.split_dir, self.split_map[split]), 'r') as f:
            self.scan_ids = [line.rstrip() for line in f.readlines()]

        # build label remapper from scan ids to nyu40 ids to consecutive numbers
        # read tsv file to get raw to nyu40 mapping (dict)
        self.raw_to_nyu40_mapping = read_label_mapping(self.label_id_tsv_path,
                                                       label_from='id', label_to='nyu40id', as_int=True)
        self.raw_to_nyu40 = np.zeros(max(self.raw_to_nyu40_mapping.keys()) + 1, dtype=np.int64)
        for key, value in self.raw_to_nyu40_mapping.items():
            self.raw_to_nyu40[key] = value
        # scannet
        self.scannet_mapping = load_class_mapping(self.scannet_classes_path)
        assert len(self.scannet_mapping) == 20
        # nyu40 -> scannet
        self.nyu40_to_scannet = np.full(shape=41, fill_value=self.ignore_value, dtype=np.int64)
        self.nyu40_to_scannet[list(self.scannet_mapping.keys())] = np.arange(len(self.scannet_mapping))
        # raw -> scannet
        self.raw_to_scannet = self.nyu40_to_scannet[self.raw_to_nyu40]
        # label mapping
        self.class_names = tuple(self.scannet_mapping.values())
        self.label_mapping = self.raw_to_scannet

        self.meta_data = self._load_dataset()
        logger = logging.getLogger(__name__)
        logger.info(str(self))

    def _load_dataset(self, cache=True):
        cache_dir = osp.join(self.root_dir, 'cache_2d')
        cache_file = osp.join(cache_dir, 'metadata_2d_{}.json'.format(self.split))
        # load cache if possible
        if cache and osp.isfile(cache_file):
            with open(cache_file, 'r') as f:
                meta_data = json.load(f)
        else:
            meta_data = []
            for scan_id in tqdm(self.scan_ids):
                scan_dir = osp.join(self.root_dir, self.scans_name.format(*self.resize), scan_id)
                color_paths = natsort.natsorted(glob.glob(osp.join(scan_dir, 'color', '*.png')))
                exclude_ids = self.exclude_frames.get(scan_id, [])
                frame_ids = [osp.splitext(osp.basename(x))[0] for x in color_paths]
                frame_ids = [x for x in frame_ids if x not in exclude_ids]
                for frame_id in frame_ids:
                    meta_data.append({
                        'scan_id': scan_id,
                        'frame_id': frame_id,
                    })
            if cache:
                os.makedirs(cache_dir, exist_ok=True)
                with open(cache_file, 'w') as f:
                    json.dump(meta_data, f)
        if self.subsample:
            meta_data = meta_data[::self.subsample]
        for data in meta_data:
            scan_dir = osp.join(self.root_dir, self.scans_name.format(*self.resize), data['scan_id'])
            frame_id = data['frame_id']
            data.update({
                # 'path/color': osp.join(scan_dir, 'color', frame_id + '.jpg'),
                'path/color': osp.join(scan_dir, 'color', frame_id + '.png'),
                'path/depth': osp.join(scan_dir, 'depth', frame_id + '.png'),
                'path/label': osp.join(scan_dir, 'label', frame_id + '.png'),
            })
        return meta_data

    def __getitem__(self, index):
        # load data
        meta_data = self.meta_data[index]
        # read as PIL image
        image = Image.open(meta_data['path/color'])
        label = Image.open(meta_data['path/label'])
        # resize
        if self.resize:
            if image.size != self.resize:
                image = image.resize(self.resize, Image.BILINEAR)
                label = label.resize(self.resize, Image.NEAREST)
        # jitter
        if self.color_jitter is not None:
            image = self.color_jitter(image)
        # horizontal flip
        if self.flip and np.random.rand() < self.flip:
            image = F.hflip(image)
            label = F.hflip(label)
        # if isinstance(image, Image.Image):
        image = np.asarray(image, dtype=np.float32) / 255.
        # if isinstance(label, Image.Image):
        label = np.asarray(label, dtype=np.int64)
        label = self.label_mapping[label].copy()
        if self.normalizer:
            mean, std = self.normalizer
            mean = np.asarray(mean, dtype=np.float32)
            std = np.asarray(std, dtype=np.float32)
            image = (image - mean) / std
        if self.to_tensor:
            image = F.to_tensor(image)  # convert to tensor and transpose
            label = torch.as_tensor(label, dtype=torch.int64)
        # return a output dictionary
        return {
            'image': image,
            'seg_label': label,
        }

    def __len__(self):
        return len(self.meta_data)

    def __str__(self):
        base_str = '{:s}: {} classes with {} images'.format(self.__class__.__name__, len(self.class_names), len(self))
        extra_str = ', '.join([x + '=' + str(getattr(self, x)) for x in ['resize', 'normalizer', 'flip', 'color_jitter']])
        return base_str + '\n' + extra_str


def test():
    from mvpnet.utils.plt_util import imshows
    dataset = ScanNet2D(osp.join(_CUR_DIR, '../../data/ScanNet'), 'val',
                        resize=(640, 480),
                        # flip=1.0,
                        # color_jitter=(0.4, 0.4, 0.4),
                        # normalizer=([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                        )
    for i in range(0, len(dataset), 1):
        data = dataset[i]
        image = data['image']
        label = data['seg_label']
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                print(k, v.shape, v.dtype)
            else:
                print(k, v)
        imshows([image, label])
