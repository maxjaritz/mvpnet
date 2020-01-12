import os
import glob
import subprocess

scannet_root = os.path.expanduser('/datasets/ScanNet')
glob_path = os.path.join(scannet_root, 'scans', '*')
scene_paths = sorted(glob.glob(glob_path))

for i, scene_path in enumerate(scene_paths):
    print('[{}/{}]'.format(i + 1, len(scene_paths)))
    os.chdir(scene_path)
    subprocess.call(['unzip', '-o', '*label.zip'])
