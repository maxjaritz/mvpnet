import os.path as osp
import numpy as np
import pickle
import os
import sys

# Assume that the script is run at the root directory
_ROOT_DIR = os.path.abspath(osp.dirname(__file__) + '/..')
sys.path.insert(0, _ROOT_DIR)

from mvpnet.data.scannet_2d import ScanNet2D

VALID_CLASS_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]


def compute_3d_weights():
    pkl_path = osp.expanduser('/home/docker_user/workspace/mvpnet_private/data/ScanNet/cache_rgbd/scannetv2_train.pkl')

    label_weights = np.zeros(41)
    # load scan data
    with open(pkl_path, 'rb') as fid:
        pickle_data = pickle.load(fid)
    for data_dict in pickle_data:
        label = data_dict['seg_label']
        label_weights += np.bincount(label, minlength=41)

    label_weights = label_weights.astype(np.float32)[VALID_CLASS_IDS]
    label_weights = label_weights / np.sum(label_weights)
    label_log_weights = 1 / np.log(1.2 + label_weights)

    np.savetxt('scannetv2_train_3d_log_weights_20_classes.txt', label_log_weights)


def compute_2d_weights():
    dataset = ScanNet2D('data/ScanNet', 'train', resize=(160, 120))
    label_weights = np.zeros(20)
    for i in range(0, len(dataset), 100):
        if (i % 10000) == 0:
            print('{}/{}'.format(i, len(dataset)))
        data = dataset[i]
        # image = data['image']
        label = data['seg_label'].flatten()
        label_weights += np.bincount(label[label != -100], minlength=20)

    label_weights = label_weights.astype(np.float32)
    label_weights = label_weights / np.sum(label_weights)
    label_log_weights = 1 / np.log(1.2 + label_weights)

    np.savetxt('scannetv2_train_2d_log_weights_20_classes.txt', label_log_weights)


if __name__ == '__main__':
    compute_2d_weights()
