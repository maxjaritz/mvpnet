import warnings
import math
import numpy as np
from scipy.spatial.transform import Rotation
import torch


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, **data_dict):
        for t in self.transforms:
            data_dict = t(**data_dict)
        return data_dict

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ToTensor(object):
    """Convert array to tensor"""
    _fp32_fields = ('points', 'normal', 'feature',)
    _int64_fields = ('seg_label',)
    _bool_fields = ('mask',)

    def __call__(self, **data_dict):
        for k, v in data_dict.items():
            if k in self._fp32_fields:
                data_dict[k] = torch.tensor(v, dtype=torch.float32)
            elif k in self._int64_fields:
                data_dict[k] = torch.tensor(v, dtype=torch.int64)
            elif k in self._bool_fields:
                data_dict[k] = torch.tensor(v, dtype=torch.bool)
            else:
                warnings.warn('Field({}) is not converted to tensor.'.format(k))
        return data_dict


class Transpose(object):
    """Transpose data to NCW/NCHW format for pytorch"""
    _fields = ('points', 'normal', 'feature',)

    def __call__(self, **data_dict):
        for k in self._fields:
            if k in data_dict:
                v = data_dict[k]
                if isinstance(v, np.ndarray):
                    assert v.ndim == 2
                    data_dict[k] = v.transpose()
                elif isinstance(v, torch.Tensor):
                    assert v.dim() == 2
                    data_dict[k] = v.transpose(0, 1)
                else:
                    raise TypeError('Wrong type {} to transpose.'.format(type(v).__name__))
        return data_dict


class RandomRotate(object):
    """Rotate along an axis by a random angle"""
    _fields = ('points', 'normal',)

    def __init__(self, axis, low=-math.pi, high=math.pi):
        self.axis = np.array(axis, dtype=np.float32)
        self.low = low
        self.high = high

    def get_rotation(self):
        angle = np.random.uniform(low=self.low, high=self.high)
        rot = Rotation.from_rotvec(angle * self.axis)
        return rot.as_dcm().astype(np.float32)

    def __call__(self, **data_dict):
        rot_mat = self.get_rotation()
        for k in self._fields:
            if k in data_dict:
                v = data_dict[k]
                assert v.ndim == 2 and v.shape[1] == 3
                data_dict[k] = v @ rot_mat.T
        return data_dict


class RandomRotateZ(RandomRotate):
    """ScanNetV2 is z-axis upward."""

    def __init__(self, *args, **kwargs):
        super(RandomRotateZ, self).__init__((0., 0., 1.), *args, **kwargs)


class Sample(object):
    """Randomly sample with replacement"""
    _fields = ('points', 'normal', 'feature', 'seg_label')

    def __init__(self, nb_pts):
        self.nb_pts = nb_pts

    def __call__(self, **data_dict):
        points = data_dict['points']
        choice = np.random.randint(len(points), size=self.nb_pts, dtype=np.int64)
        for k in self._fields:
            if k in data_dict:
                v = data_dict[k]
                data_dict[k] = v[choice]
        return data_dict


class CropPad(object):
    """Crop or pad point clouds"""
    _fields = ('points', 'normal', 'feature', 'seg_label')

    def __init__(self, nb_pts):
        self.nb_pts = nb_pts

    def __call__(self, **data_dict):
        points = data_dict['points']
        # mask = np.ones(self.nb_pts, dtype=bool)
        if len(points) < self.nb_pts:
            pad = np.random.randint(len(points), size=self.nb_pts - len(points))
            choice = np.hstack([np.arange(len(points)), pad])
            # mask[len(points):] = 0
        else:
            choice = np.random.choice(len(points), size=self.nb_pts, replace=False)
        for k in self._fields:
            if k in data_dict:
                v = data_dict[k]
                data_dict[k] = v[choice]
        # data_dict['mask'] = mask
        return data_dict


class Pad(CropPad):
    """Pad point clouds. Only for test."""

    def __call__(self, **data_dict):
        points = data_dict['points']
        if len(points) < self.nb_pts:
            pad = np.random.randint(len(points), size=self.nb_pts - len(points))
            choice = np.hstack([np.arange(len(points)), pad])
            for k in self._fields:
                if k in data_dict:
                    v = data_dict[k]
                    data_dict[k] = v[choice]
        return data_dict
