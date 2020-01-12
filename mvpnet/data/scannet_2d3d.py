import os.path as osp
import logging
import pickle

import numpy as np
from scipy.spatial.transform import Rotation
from sklearn.neighbors import NearestNeighbors
from PIL import Image
import open3d as o3d
from torch.utils.data import Dataset
from torchvision.transforms import transforms as T

_CUR_DIR = osp.dirname(__file__)
_META_DIR = osp.abspath(osp.join(_CUR_DIR, 'meta_files'))

from mvpnet.data.scannet_2d import load_class_mapping, read_label_mapping
from mvpnet.utils.chunk_util import scene2chunks_legacy


def select_frames(rgbd_overlap, num_rgbd_frames):
    selected_frames = []
    # make a copy to avoid modifying input
    rgbd_overlap = rgbd_overlap.copy()
    for i in range(num_rgbd_frames):
        # choose the RGBD frame with the maximum overlap (measured in num basepoints)
        frame_idx = rgbd_overlap.sum(0).argmax()
        selected_frames.append(frame_idx)
        # set all points covered by this frame to invalid
        rgbd_overlap[rgbd_overlap[:, frame_idx]] = False
    return selected_frames


def depth2xyz(cam_matrix, depth):
    # create xyz coordinates from image position
    v, u = np.indices(depth.shape)
    u, v = u.ravel(), v.ravel()
    uv1_points = np.stack([u, v, np.ones_like(u)], axis=1)
    xyz = (np.linalg.inv(cam_matrix[:3, :3]).dot(uv1_points.T) * depth.ravel()).T
    return xyz


class ScanNet2D3DChunks(Dataset):
    """ScanNetV2 2D-3D chunks dataset"""
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
    ignore_value = -100

    def __init__(self,
                 cache_dir,
                 image_dir,
                 split,
                 chunk_size=(1.5, 1.5),
                 chunk_thresh=0.3,
                 chunk_margin=(0.2, 0.2),
                 nb_pts=-1,
                 num_rgbd_frames=0,
                 resize=(160, 120),
                 image_normalizer=None,
                 k=3,
                 z_rot=None,
                 flip=0.0,
                 color_jitter=None,
                 to_tensor=False,
                 ):
        """

        Args:
            cache_dir (str): path to cache of 3D point clouds, 3D semantic labels and RGB-D overlap info
            image_dir (str): path to 2D images, depth maps and poses
            split:
            chunk_size (tuple): xy chunk size
            chunk_thresh (float): minimum number of labeled points within a chunk
            chunk_margin (tuple): margin to calculate ratio of labeled points within a chunk
            nb_pts (int): number of points to resample in a chunk
            num_rgbd_frames (int): number of RGB-D frames to choose
            resize (tuple): target image size
            image_normalizer (tuple, optional): (mean, std)
            k (int): k-nn unprojected neighbors of target points
            z_rot (tuple, optional): range of rotation (degree instead of rad)
            flip (float): probability to flip horizontally
            color_jitter (tuple, optional): paramters of color jitter
            to_tensor (bool): whether to convert to torch.Tensor
        """
        super(ScanNet2D3DChunks, self).__init__()

        # cache: pickle files containing point clouds, 3D labels and rgbd overlap
        self.cache_dir = cache_dir
        # includes color, depth, 2D label
        self.image_dir = image_dir

        # load split
        self.split = split
        with open(osp.join(self.split_dir, self.split_map[split]), 'r') as f:
            self.scan_ids = [line.rstrip() for line in f.readlines()]

        # ---------------------------------------------------------------------------- #
        # Build label mapping
        # ---------------------------------------------------------------------------- #
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
        # scannet -> nyu40
        self.scannet_to_nyu40 = np.array(list(self.scannet_mapping.keys()) + [0], dtype=np.int64)
        # raw -> scannet
        self.raw_to_scannet = self.nyu40_to_scannet[self.raw_to_nyu40]
        self.class_names = tuple(self.scannet_mapping.values())

        # ---------------------------------------------------------------------------- #
        # 3D
        # ---------------------------------------------------------------------------- #
        # The height / z-axis is ignored in fact.
        self.chunk_size = np.array(chunk_size, dtype=np.float32)
        self.chunk_thresh = chunk_thresh
        self.chunk_margin = np.array(chunk_margin, dtype=np.float32)
        self.nb_pts = nb_pts

        # ---------------------------------------------------------------------------- #
        # 2D
        # ---------------------------------------------------------------------------- #
        self.num_rgbd_frames = num_rgbd_frames
        self.resize = resize
        self.image_normalizer = image_normalizer

        # ---------------------------------------------------------------------------- #
        # 2D-3D
        # ---------------------------------------------------------------------------- #
        self.k = k
        if num_rgbd_frames > 0 and resize:
            depth_size = (640, 480)  # intrinsic matrix is based on 640x480 depth maps.
            self.resize_scale = (depth_size[0] / resize[0], depth_size[1] / resize[1])
        else:
            self.resize_scale = None

        # ---------------------------------------------------------------------------- #
        # Augmentation
        # ---------------------------------------------------------------------------- #
        self.z_rot = z_rot
        self.flip = flip
        self.color_jitter = T.ColorJitter(*color_jitter) if color_jitter else None
        self.to_tensor = to_tensor

        # ---------------------------------------------------------------------------- #
        # Load cache data
        # ---------------------------------------------------------------------------- #
        # import time
        # tic = time.time()
        self._load_dataset()
        # print(time.time() - tic)

        logger = logging.getLogger(__name__)
        logger.info(str(self))

    def _load_dataset(self):
        with open(osp.join(self.cache_dir, 'scannetv2_{}.pkl'.format(self.split)), 'rb') as f:
            cache_data = pickle.load(f)
        self.data = cache_data
        self.scan_ids = [scan['scan_id'] for scan in cache_data]
        self.total_frames = sum(len(x['frame_ids']) for x in cache_data)

    def _get_paths(self, scan_dir):
        return {
            'color': osp.join(scan_dir, 'color', '{}.png'),
            # 'color': osp.join(scan_dir, 'color', '{}.jpg'),
            'depth': osp.join(scan_dir, 'depth', '{}.png'),
            'pose': osp.join(scan_dir, 'pose', '{}.txt'),
            '2d_label': osp.join(scan_dir, 'label', '{}.png'),
        }

    def get_rgbd_data(self, data_dict, chunk_points, chunk_box, chunk_mask):
        scan_id = data_dict['scan_id']
        scan_dir = osp.join(self.image_dir, scan_id)
        paths = self._get_paths(scan_dir)

        # load data
        # nb: number of base points; np: number of all points; nc: number of chunk points
        # nf: number of frames; nbc: number of base points in the chunk
        base_point_ind = data_dict['base_point_ind'].astype(np.int64)  # (nb,)
        base_point_mask = np.zeros_like(chunk_mask)  # (np,)
        base_point_mask[base_point_ind] = True  # (np,)
        base_point_mask = np.logical_and(base_point_mask, chunk_mask)  # (np,)
        pointwise_rgbd_overlap = data_dict['pointwise_rgbd_overlap'].astype(np.bool_)  # (nb, nf)
        chunk_basepoint_rgbd_overlap = pointwise_rgbd_overlap[base_point_mask[base_point_ind]]  # (nbc, nf)
        frame_ids = data_dict['frame_ids']
        cam_matrix = data_dict['cam_matrix'].astype(np.float32)
        # adapt camera matrix
        if self.resize_scale is not None:
            cam_matrix[0] /= self.resize_scale[0]
            cam_matrix[1] /= self.resize_scale[1]

        # # visualize basepoints
        # from mvpnet.utils.o3d_util import draw_point_cloud
        # from mvpnet.utils.visualize import label2color
        # pts_vis = draw_point_cloud(chunk_points, label2color(chunk_seg_label))
        # base_pts_vis = draw_point_cloud(points[base_point_mask], [1., 0., 0.])
        # o3d.visualization.draw_geometries([base_pts_vis, pts_vis])

        # greedy choose frames
        selected_frames = select_frames(chunk_basepoint_rgbd_overlap, self.num_rgbd_frames)
        selected_frames = [frame_ids[x] for x in selected_frames]

        # process frames
        image_list = []
        image_xyz_list = []
        image_mask_list = []
        for i, frame_id in enumerate(selected_frames):
            # load image, depth, pose
            image = Image.open(paths['color'].format(frame_id))
            depth = Image.open(paths['depth'].format(frame_id))
            pose = np.loadtxt(paths['pose'].format(frame_id), dtype=np.float32)

            # resize
            if self.resize:
                if not image.size == self.resize:
                    # check if we do not enlarge downsized images
                    assert image.size[0] > self.resize[0] and image.size[1] > self.resize[1]
                    image = image.resize(self.resize, Image.BILINEAR)
                    depth = depth.resize(self.resize, Image.NEAREST)

            # color jitter
            if self.color_jitter is not None:
                image = self.color_jitter(image)

            # normalize image
            image = np.asarray(image, dtype=np.float32) / 255.
            if self.image_normalizer:
                mean, std = self.image_normalizer
                mean = np.asarray(mean, dtype=np.float32)
                std = np.asarray(std, dtype=np.float32)
                image = (image - mean) / std
            image_list.append(image)

            # rescale depth
            depth = np.asarray(depth, dtype=np.float32) / 1000.

            # inverse perspective transformation
            image_xyz = depth2xyz(cam_matrix, depth)  # (h * w, 3)
            # find valid depth
            image_mask = image_xyz[:, 2] > 0  # (h * w)
            # camera -> world
            image_xyz = np.matmul(image_xyz, pose[:3, :3].T) + pose[:3, 3]

            # # visualize unprojected point clouds
            # from mvpnet.utils.o3d_util import draw_point_cloud
            # pcd_xyz = draw_point_cloud(image_xyz[image_mask], [0., 1., 0.])
            # pcd_scene = draw_point_cloud(points, [1., 0., 0.])
            # o3d.visualization.draw_geometries([pcd_xyz, pcd_scene])

            if not np.any(image_mask):
                print('Invalid depth map for frame {} of scan {}.'.format(frame_id, scan_id))

            # set invalid flags outsides the chunk
            if chunk_box is not None:
                margin = 0.1
                in_chunk_mask = np.logical_and.reduce(
                    (image_xyz[:, 0] > chunk_box[0] - margin,
                     image_xyz[:, 0] < chunk_box[2] + margin,
                     image_xyz[:, 1] > chunk_box[1] - margin,
                     image_xyz[:, 1] < chunk_box[3] + margin))
                image_mask = np.logical_and(image_mask, in_chunk_mask)

            image_xyz_list.append(image_xyz)
            image_mask_list.append(image_mask)

        # post-process, especially for horizontal flip
        image_ind_list = []
        for i in range(self.num_rgbd_frames):
            h, w, _ = image_list[i].shape
            # reshape
            image_xyz_list[i] = image_xyz_list[i].reshape([h, w, 3])
            image_mask_list[i] = image_mask_list[i].reshape([h, w])
            if self.flip and np.random.rand() < self.flip:
                image_list[i] = np.fliplr(image_list[i])
                image_xyz_list[i] = np.fliplr(image_xyz_list[i])
                image_mask_list[i] = np.fliplr(image_mask_list[i])
            image_mask = image_mask_list[i]
            image_ind = np.nonzero(image_mask.ravel())[0]
            if image_ind.size > 0:
                image_ind_list.append(image_ind + i * h * w)
            else:
                image_ind_list.append([])

        images = np.stack(image_list, axis=0)  # (nv, h, w, 3)
        image_xyz_valid = np.concatenate([image_xyz[image_mask] for image_xyz, image_mask in
                                          zip(image_xyz_list, image_mask_list)], axis=0)
        image_ind_all = np.hstack(image_ind_list)  # (n_valid,)

        # Find k-nn in dense point clouds for each sparse point
        nbrs = NearestNeighbors(n_neighbors=self.k, algorithm='ball_tree').fit(image_xyz_valid)
        _, knn_indices = nbrs.kneighbors(chunk_points)  # (nc, 3)
        # remap to pixel index
        knn_indices = image_ind_all[knn_indices]

        out_dict = {
            'images': images.astype(np.float32, copy=False),
            'image_xyz': np.stack(image_xyz_list, axis=0).astype(np.float32, copy=False),  # (nv, h, w, 3)
            'image_mask': np.stack(image_mask_list, axis=0).astype(np.bool_, copy=False),  # (nv, h, w)
            'knn_indices': knn_indices.astype(np.int64, copy=False),  # (nc, 3)
        }
        return out_dict

    def __getitem__(self, index):
        data_dict = self.data[index]
        scan_id = data_dict['scan_id']
        assert scan_id == self.scan_ids[index], 'Mismatch scan_id: {} vs {}.'.format(scan_id, self.scan_ids[index])

        # ---------------------------------------------------------------------------- #
        # Load 3D data
        # ---------------------------------------------------------------------------- #
        # Note that astype will copy data.
        points = data_dict['points'].astype(np.float32)
        seg_label = data_dict['seg_label']
        seg_label = self.nyu40_to_scannet[seg_label].astype(np.int64)

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
            chunk_min = np.min(points[:, :2], axis=0)
            chunk_max = np.max(points[:, :2], axis=0)
            chunk_points = points
            chunk_seg_label = seg_label
            chunk_mask = np.ones_like(seg_label, dtype=bool)
            print('No valid chunk found in scene {}. '
                  'Return all points({:d}).'.format(scan_id, points.shape[0]))

        # Resample points to a fixed number
        chunk_nb_pts = len(chunk_points)
        if chunk_nb_pts < self.nb_pts:
            pad = np.random.randint(chunk_nb_pts, size=self.nb_pts - chunk_nb_pts)
            crop_pad_choice = np.hstack([np.arange(chunk_nb_pts), pad])
        else:
            crop_pad_choice = np.random.choice(chunk_nb_pts, size=self.nb_pts, replace=False)
        chunk_points = chunk_points[crop_pad_choice]
        chunk_seg_label = chunk_seg_label[crop_pad_choice]

        out_dict = {
            'points': chunk_points,
            'seg_label': chunk_seg_label,
        }

        # ---------------------------------------------------------------------------- #
        # On-the-fly find frames to maximally cover the chunk
        # ---------------------------------------------------------------------------- #
        if self.num_rgbd_frames > 0:
            # bounding box: (x_min, y_min, x_max, y_max)
            chunk_box = np.hstack([chunk_min - self.chunk_margin, chunk_max + self.chunk_margin])
            out_dict_rgbd = self.get_rgbd_data(data_dict, chunk_points, chunk_box, chunk_mask)
            out_dict.update(out_dict_rgbd)

        # ---------------------------------------------------------------------------- #
        # Data augmentation
        # ---------------------------------------------------------------------------- #
        if self.z_rot:
            # Note that Rotation currently does not process dtype well.
            angle = np.random.uniform(low=self.z_rot[0], high=self.z_rot[1])
            Rot = Rotation.from_euler('z', angle, degrees=True)
            out_dict['points'] = Rot.apply(out_dict['points'])
            out_dict['points'] = out_dict['points'].astype(dtype=np.float32, copy=False)
            if 'image_xyz' in out_dict:
                image_xyz = out_dict['image_xyz']
                out_dict['image_xyz'] = Rot.apply(image_xyz.reshape([-1, 3])).reshape(image_xyz.shape)
                out_dict['image_xyz'] = out_dict['image_xyz'].astype(dtype=np.float32, copy=False)

        if self.to_tensor:
            out_dict['points'] = out_dict['points'].T  # (3, nc)
            if 'images' in out_dict:
                out_dict['images'] = np.moveaxis(out_dict['images'], -1, 1)  # (nv, 3, h, w)

        return out_dict

    def __len__(self):
        return len(self.scan_ids)

    def __str__(self):
        base_str = '{:s}: {} classes with {} scenes and {} frames.'.format(
            self.__class__.__name__, len(self.class_names), len(self.scan_ids), self.total_frames)
        attr_list = ['chunk_size', 'chunk_thresh', 'chunk_margin', 'nb_pts', 'num_rgbd_frames', 'resize',
                     'image_normalizer', 'k', 'z_rot', 'flip']
        extra_str = ', '.join([x + '=' + str(getattr(self, x)) for x in attr_list])
        return base_str + '\n' + extra_str


def test_ScanNet2D3DChunks():
    from mvpnet.utils.o3d_util import visualize_point_cloud
    from mvpnet.utils.visualize import visualize_labels
    cache_dir = osp.join('/home/docker_user/workspace/mvpnet_private/data/ScanNet/cache_rgbd')
    image_dir = osp.join('/home/docker_user/workspace/mvpnet_private/data/ScanNet/scans_resize_160x120')

    np.random.seed(0)
    dataset = ScanNet2D3DChunks(cache_dir=cache_dir,
                                image_dir=image_dir,
                                split='val',
                                nb_pts=8192,
                                num_rgbd_frames=3,
                                # color_jitter=(0.5, 0.5, 0.5),
                                # flip=0.5,
                                # z_rot=(-180, 180),
                                )
    print(dataset)
    for i in range(len(dataset)):
        data = dataset[i]
        points = data['points']
        seg_label = data['seg_label']
        # colors = data.get('feature', None)
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                print(k, v.shape, v.dtype)
            else:
                print(k, v)
        # visualize_labels(points, seg_label, style='scannet')

        images = data['images']
        image_xyz = data['image_xyz']
        image_mask = data['image_mask']
        image_xyz_valid = image_xyz[image_mask]
        visualize_point_cloud(image_xyz_valid)

        knn_indices = data['knn_indices']
        # visualize_point_cloud(image_xyz_valid.reshape(-1, 3)[knn_indices[:, 1]],
        #                       images[image_mask].reshape(-1, 3)[knn_indices[:, 1]],
        #                       )
        visualize_point_cloud(image_xyz.reshape(-1, 3)[knn_indices[:, 2]],
                              images.reshape(-1, 3)[knn_indices[:, 2]],
                              )


class ScanNet2D3DChunksTest(ScanNet2D3DChunks):
    def __init__(self,
                 cache_dir,
                 image_dir,
                 split,
                 chunk_size=(1.5, 1.5),
                 chunk_stride=0.5,
                 chunk_thresh=1000,
                 chunk_margin=(0.2, 0.2),
                 nb_pts=-1,
                 num_rgbd_frames=0,
                 resize=(160, 120),
                 image_normalizer=None,
                 k=3,
                 to_tensor=False,
                 ):
        super(ScanNet2D3DChunksTest, self).__init__(
            cache_dir,
            image_dir,
            split,
            chunk_size=chunk_size,
            chunk_thresh=chunk_thresh,
            chunk_margin=chunk_margin,
            nb_pts=nb_pts,
            num_rgbd_frames=num_rgbd_frames,
            resize=resize,
            image_normalizer=image_normalizer,
            k=k,
            to_tensor=to_tensor,
        )
        self.chunk_stride = chunk_stride

    def __getitem__(self, index):
        data_dict = self.data[index]
        scan_id = data_dict['scan_id']
        assert scan_id == self.scan_ids[index], 'Mismatch scan_id: {} vs {}.'.format(scan_id, self.scan_ids[index])

        # ---------------------------------------------------------------------------- #
        # Load 3D data
        # ---------------------------------------------------------------------------- #
        # Note that astype will copy data.
        points = data_dict['points'].astype(np.float32)
        seg_label = data_dict.get('seg_label', None)
        if seg_label is not None:
            seg_label = self.nyu40_to_scannet[seg_label].astype(np.int64)

        # ---------------------------------------------------------------------------- #
        # Sliding chunks
        # ---------------------------------------------------------------------------- #
        chunk_indices, chunk_boxes = scene2chunks_legacy(points,
                                                         chunk_size=self.chunk_size,
                                                         stride=self.chunk_stride,
                                                         thresh=self.chunk_thresh,
                                                         margin=self.chunk_margin,
                                                         return_bbox=True)

        out_dict_list = []
        for chunk_ind, chunk_box in zip(chunk_indices, chunk_boxes):
            chunk_points = points[chunk_ind].copy()
            # Resample points to a fixed number
            chunk_nb_pts = len(chunk_points)
            if self.nb_pts <= 0:
                crop_pad_choice = np.arange(chunk_nb_pts)
            elif chunk_nb_pts < self.nb_pts:
                pad = np.random.randint(chunk_nb_pts, size=self.nb_pts - chunk_nb_pts)
                crop_pad_choice = np.hstack([np.arange(chunk_nb_pts), pad])
            else:
                crop_pad_choice = np.random.choice(chunk_nb_pts, size=self.nb_pts, replace=False)
            chunk_points = chunk_points[crop_pad_choice]
            out_dict = {
                'points': chunk_points,
                'chunk_ind': chunk_ind,
            }
            if seg_label is not None:
                out_dict['seg_label'] = seg_label[chunk_ind][crop_pad_choice]

            # ---------------------------------------------------------------------------- #
            # On-the-fly find frames to maximally cover the chunk
            # ---------------------------------------------------------------------------- #
            if self.num_rgbd_frames > 0:
                chunk_mask = np.zeros([len(points)], dtype=bool)
                chunk_mask[chunk_ind] = True
                out_dict_rgbd = self.get_rgbd_data(data_dict, chunk_points, chunk_box[[0, 1, 3, 4]], chunk_mask)
                out_dict.update(out_dict_rgbd)

            if self.to_tensor:
                out_dict['points'] = out_dict['points'].T  # (3, nc)
                if 'images' in out_dict:
                    out_dict['images'] = np.moveaxis(out_dict['images'], -1, 1)  # (nv, 3, h, w)
            out_dict_list.append(out_dict)

        return out_dict_list


def test_ScanNet2D3DChunksTest():
    from mvpnet.utils.o3d_util import visualize_point_cloud
    from mvpnet.utils.visualize import visualize_labels
    cache_dir = osp.join('/home/jiayuan/Projects/mvpnet_private/data/ScanNet/cache_rgbd')
    image_dir = osp.join('/home/jiayuan/Projects/mvpnet_private/data/ScanNet/scans_resize_160x120')

    np.random.seed(0)
    dataset = ScanNet2D3DChunksTest(cache_dir=cache_dir,
                                    image_dir=image_dir,
                                    split='val',
                                    nb_pts=8192,
                                    num_rgbd_frames=3,
                                    )
    print(dataset)
    for i in range(len(dataset)):
        data_list = dataset[i]
        for data in data_list:
            points = data['points']
            seg_label = data['seg_label']
            # colors = data.get('feature', None)
            for k, v in data.items():
                if isinstance(v, np.ndarray):
                    print(k, v.shape, v.dtype)
                else:
                    print(k, v)
            visualize_labels(points, seg_label, style='scannet')

            images = data['images']
            image_xyz = data['image_xyz']
            image_mask = data['image_mask']
            image_xyz_valid = image_xyz[image_mask]
            visualize_point_cloud(image_xyz_valid)

            knn_indices = data['knn_indices']
            # visualize_point_cloud(image_xyz_valid.reshape(-1, 3)[knn_indices[:, 1]],
            #                       images[image_mask].reshape(-1, 3)[knn_indices[:, 1]],
            #                       )
            visualize_point_cloud(image_xyz.reshape(-1, 3)[knn_indices[:, 2]],
                                  images.reshape(-1, 3)[knn_indices[:, 2]],
                                  )


if __name__ == '__main__':
    test_ScanNet2D3DChunks()
