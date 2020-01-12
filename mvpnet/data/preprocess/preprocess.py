"""Dump 3D data of ScanNetV2 to pickle files and computes the RGBD overlaps with the whole scene point cloud.
References: https://github.com/ScanNet/ScanNet/tree/master/BenchmarkScripts
"""

import os
import os.path as osp
import pickle
import time
import argparse
from functools import partial
import glob

import numpy as np
import natsort
from plyfile import PlyData
from PIL import Image
import open3d as o3d

# DATA_DIR = '/datasets_local/ScanNet'
# SCAN_DIR = 'scans_resize_160x120'
# META_DIR = '/home/docker_user/workspace/mvpnet_private/mvpnet/data/meta_files'

# DATA_DIR = '/data/dataset/ScanNet'
# SCAN_DIR = 'scans'
DATA_DIR = '/home/jiayuan/Projects/mvpnet_private/data/ScanNet'
SCAN_DIR = 'scans_resize_160x120'
SCAN_TEST_DIR = 'scans_test'
META_DIR = '/home/jiayuan/Projects/mvpnet_private/mvpnet/data/meta_files'

SEG_CLASS_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
# INST_CLASS_IDS = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
_DEBUG = False
# _DEBUG = True

# exclude some frames with problematic data (e.g. depth frames with zeros everywhere or unreadable labels)
exclude_frames = {
    'scene0243_00': ['1175', '1176', '1177', '1178', '1179', '1180', '1181', '1182', '1183', '1184'],
    'scene0538_00': ['1925', '1928', '1929', '1931', '1932', '1933'],
    'scene0639_00': ['442', '443', '444'],
    'scene0299_01': ['1512'],
}


# ----------------------------------------------------------------------------- #
# Helpers
# ----------------------------------------------------------------------------- #
def get_paths(scan_dir, scan_id):
    return {
        'point_cloud': osp.join(scan_dir, scan_id + '_vh_clean_2.ply'),
        'label_3d': osp.join(scan_dir, scan_id + '_vh_clean_2.labels.ply'),
        'segment_data': osp.join(scan_dir, scan_id + '_vh_clean_2.0.010000.segs.json'),
        # 'aggregation_data': osp.join(scan_dir, scan_id + '.aggregation.json'),
        'aggregation_data': osp.join(scan_dir, scan_id + '_vh_clean.aggregation.json'),
        'color': osp.join(scan_dir, 'color', '{}.jpg'),
        'depth': osp.join(scan_dir, 'depth', '{}.png'),
        'pose': osp.join(scan_dir, 'pose', '{}.txt'),
        'label_2d': osp.join(scan_dir, 'label', '{}.png'),
        'intrinsics_depth': osp.join(scan_dir, 'intrinsic', 'intrinsic_depth.txt'),
        'scene_info_txt': osp.join(scan_dir, scan_id + '.txt'),
        'label_id_tsv': osp.join(META_DIR, 'scannetv2-labels.combined.tsv'),
    }


def read_pc_from_ply(filename, return_color=False, return_label=False):
    """Read point clouds from ply files"""
    ply_data = PlyData.read(filename)
    vertex = ply_data['vertex']
    x = np.asarray(vertex['x'])
    y = np.asarray(vertex['y'])
    z = np.asarray(vertex['z'])
    points = np.stack([x, y, z], axis=1)
    pc = {'points': points}
    if return_color:
        r = np.asarray(vertex['red'])
        g = np.asarray(vertex['green'])
        b = np.asarray(vertex['blue'])
        colors = np.stack([r, g, b], axis=1)
        pc['colors'] = colors
    if return_label:
        label = np.asarray(vertex['label'])
        pc['label'] = label
    return pc


def unproject(k, depth_map, mask=None):
    if mask is None:
        # only consider points where we have a depth value
        mask = depth_map > 0
    # create xy coordinates from image position
    v, u = np.indices(depth_map.shape)
    v = v[mask]
    u = u[mask]
    depth = depth_map[mask].ravel()
    uv1_points = np.stack([u, v, np.ones_like(u)], axis=1)
    points_3d_xyz = (np.linalg.inv(k[:3, :3]).dot(uv1_points.T) * depth).T
    return points_3d_xyz


def compute_rgbd_knn(frame_ids, cam_matrix, paths, whole_scene_pts,
                     num_base_pts=2000, resize=(80, 60)):
    # choose m base points
    base_point_ind = np.random.choice(len(whole_scene_pts), num_base_pts, replace=False)
    base_pts = whole_scene_pts[base_point_ind]

    # initialize output
    overlaps = np.zeros([len(base_point_ind), len(frame_ids)], dtype=bool)

    # visualize base points
    if _DEBUG:
        from mvpnet.utils.o3d_util import draw_point_cloud
        pts_vis = draw_point_cloud(whole_scene_pts, colors=[0., 1., 0.])
        base_pts_vis = draw_point_cloud(base_pts, colors=[1., 0., 0.])
        o3d.visualization.draw_geometries([pts_vis, base_pts_vis])

    # build kd tree for base points
    base_pts_pc = o3d.geometry.PointCloud()
    base_pts_pc.points = o3d.utility.Vector3dVector(base_pts)
    pcd_tree = o3d.geometry.KDTreeFlann(base_pts_pc)

    if resize:
        # Note that we may use 160x120 depth maps; however, camera matrix here is irrelevant to that.
        # adjust intrinsic matrix
        depth_map_size = (640, 480)
        cam_matrix = cam_matrix.copy()  # avoid overwriting
        cam_matrix[0] /= depth_map_size[0] / resize[0]
        cam_matrix[1] /= depth_map_size[1] / resize[1]

    last_time = time.time()
    for i, frame_id in enumerate(frame_ids):
        if (i + 1 % 1000) == 0:
            now = time.time()
            print('[{}/{}] time: {:.2f}'.format(i + 1, len(frame_ids), now - last_time))
            last_time = now

        # load pose
        pose = np.loadtxt(paths['pose'].format(frame_id))
        if np.any(np.isinf(pose)):
            print('Skipping frame {}, because pose is not valid.'.format(frame_id))
            continue

        # load depth map
        depth = Image.open(paths['depth'].format(frame_id))
        if resize:
            depth = depth.resize(resize, Image.NEAREST)
        depth = np.asarray(depth, dtype=np.float32) / 1000.

        # un-project point cloud from depth map
        unproj_pts = unproject(cam_matrix, depth)

        # apply pose to unprojected points
        unproj_pts = pose[:3, :3].dot(unproj_pts[:, :3].T).T + pose[:3, 3]

        # for each point of unprojected point cloud find nearest neighbor (only one!) in whole scene point cloud
        for j in range(len(unproj_pts)):
            # find a neighbor that is at most 1cm away
            found, idx_point, dist = pcd_tree.search_hybrid_vector_3d(unproj_pts[j, :3], 0.1, 1)
            if found:
                overlaps[idx_point, i] = True

        # visualize
        if _DEBUG:
            from mvpnet.utils.o3d_util import draw_point_cloud
            # pts_vis = draw_point_cloud(whole_scene_pts)
            base_pts_vis = draw_point_cloud(base_pts, colors=[1., 0., 0.])
            overlap_base_pts_vis = draw_point_cloud(base_pts[overlaps[:, i]], colors=[0., 1., 0.])
            unproj_pts_vis = draw_point_cloud(unproj_pts, colors=[0., 0., 1.])
            # o3d.visualization.draw_geometries([overlap_base_pts_vis, unproj_pts_vis, base_pts_vis, pts_vis])
            o3d.visualization.draw_geometries([base_pts_vis, unproj_pts_vis, overlap_base_pts_vis])

    return base_point_ind, overlaps


# ----------------------------------------------------------------------------- #
# Worker function
# ----------------------------------------------------------------------------- #
def process_scan_3d_sem_seg(scan_id, is_test=False, compute_rgbd_overlap=False, output_dir=None):
    """Convert raw data into a dictionary

    The data structure is as follows:
        - scan_id (str)
        - points (np.float32): (n, 3)
        - colors (np.uint8): (n, 3), rgb
        - seg_label (np.uint16): (n,), nyu40 style

    """
    print('Processing {}'.format(scan_id))
    start_time = time.time()
    data_dict = {'scan_id': scan_id}
    # set random seed
    seed = int(scan_id[5:9] + scan_id[10:12])
    # print(seed)
    np.random.seed(seed)

    #######################################
    # load data
    #######################################
    if not is_test:
        scan_dir = osp.join(DATA_DIR, SCAN_DIR, scan_id)
    else:
        scan_dir = osp.join(DATA_DIR, SCAN_TEST_DIR, scan_id)
    paths = get_paths(scan_dir, scan_id)

    # load 3D data
    pc = read_pc_from_ply(paths['point_cloud'], return_color=True)
    data_dict['points'] = pc['points']  # float32
    data_dict['colors'] = pc['colors']  # uint8

    if not is_test:
        pc_label = read_pc_from_ply(paths['label_3d'], return_label=True)
        try:
            np.testing.assert_allclose(pc['points'], pc_label['points'])
        except AssertionError:
            print(scan_id, 'mismatch points and labels.')

        # scene0270_00 and scene0384_00 have bad labels (50 and 149)
        label_3d = pc_label['label']  # uint16
        bad_label_ind = np.logical_or(label_3d < 0, label_3d > 40)
        if bad_label_ind.any():
            print(scan_id, 'bad labels: {}.'.format(np.unique(label_3d[bad_label_ind], return_counts=True)))
            label_3d[bad_label_ind] = 0
        data_dict['seg_label'] = label_3d

    if compute_rgbd_overlap:
        # get frames in scan
        glob_path = osp.join(scan_dir, 'color', '*')
        cam_matrix = np.loadtxt(paths['intrinsics_depth'], dtype=np.float32)
        color_paths = natsort.natsorted(glob.glob(glob_path))
        exclude_ids = exclude_frames.get(scan_id, [])
        frame_ids = [osp.splitext(osp.basename(x))[0] for x in color_paths]
        frame_ids = [x for x in frame_ids if x not in exclude_ids]
        if not frame_ids:
            print('WARNING: No frames found, check glob path {}'.format(glob_path))

        base_point_ind, pointwise_rgbd_overlap = compute_rgbd_knn(frame_ids, cam_matrix, paths, data_dict['points'])
        data_dict.update({
            'base_point_ind': base_point_ind,
            'pointwise_rgbd_overlap': pointwise_rgbd_overlap,
            'frame_ids': frame_ids,
            'cam_matrix': cam_matrix,
        })

    #######################################
    # save data
    #######################################
    if output_dir is not None:
        output_path = osp.join(output_dir, '{}.pkl'.format(scan_id))
        print('Save to {}'.format(osp.abspath(output_path)))
        with open(output_path, 'wb') as f:
            pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(scan_id, '{:.2f}s'.format(time.time() - start_time))
    return data_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output-dir', type=str, required=True, help='output directory')
    parser.add_argument('-n', '--num-workers', default=16, type=int, help='number of workers')
    parser.add_argument('-s', '--split', required=True, type=str, help='split[train/val/test]')
    parser.add_argument('--rgbd', action='store_true', help='compute RGBD overlaps for MVPNet')
    args = parser.parse_args()

    split_filename = osp.join(META_DIR, 'scannetv2_{:s}.txt'.format(args.split))
    with open(split_filename, 'r') as f:
        scan_ids = [line.rstrip() for line in f]
    is_test = (args.split == 'test')
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Debug
    if _DEBUG:
        for scan_id in scan_ids:
            process_scan_3d_sem_seg(scan_id, compute_rgbd_overlap=args.rgbd, is_test=is_test)
        exit(0)

    # Multiprocessing
    import multiprocessing as mp
    p = mp.Pool(args.num_workers)
    func = partial(process_scan_3d_sem_seg, compute_rgbd_overlap=args.rgbd, is_test=is_test)
    res = p.map(func, scan_ids, chunksize=1)
    p.close()
    p.join()

    if args.output_dir is not None:
        output_path = osp.join(output_dir, 'scannetv2_{}.pkl'.format(args.split))
        print('Save to {}'.format(osp.abspath(output_path)))
        with open(output_path, 'wb') as f:
            pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
