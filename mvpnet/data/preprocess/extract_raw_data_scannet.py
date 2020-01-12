from __future__ import print_function

import glob
import os
import os.path as osp
import subprocess
import multiprocessing as mp

# path to reader script
sens_reader_path = osp.join('SensReader', 'reader.py')

# glob all sens meta_files
# scannet_root and output_dir can be equal. We suggest putting sens files on HDD and extract onto a SSD.
scannet_root = "/datasets_hdd/ScanNet"
output_dir = "/datasets_ssd/ScanNet"
scans_dir = 'scans'
# scans_dir = 'scans_test'
glob_path = os.path.join(scannet_root, scans_dir, '*', '*.sens')
sens_paths = sorted(glob.glob(glob_path))


def extract(a):
    i, sens_path = a
    rest, sens_filename = os.path.split(sens_path)
    scan_id = os.path.split(rest)[1]
    output_path = os.path.join(output_dir, scans_dir, scan_id)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    print('Processing file {}/{}: {} '.format(i + 1, len(sens_paths), sens_filename))
    process = subprocess.Popen(['python', sens_reader_path,
                                '--filename', sens_path,
                                '--output_path', output_path,
                                '--export_depth_images',
                                '--export_color_images',
                                '--export_poses',
                                '--export_intrinsics']
                               )
    process.wait()


# # without multiprocessing
# for i in range(len(sens_paths)):
#     extract((i, sens_paths[i]))


# with multiprocessing
p = mp.Pool(24)
p.map(extract, [(i, sens_paths[i]) for i in range(len(sens_paths))], chunksize=1)
p.close()
p.join()
