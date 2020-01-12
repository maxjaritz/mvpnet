import os
import os.path as osp
import time
import socket
import numpy as np
import sys
import scipy.special

scannet_to_nyu40 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39, 0])

# Assume that the script is run at the root directory
_ROOT_DIR = os.path.abspath(osp.dirname(__file__) + '/..')
sys.path.insert(0, _ROOT_DIR)

from mvpnet.data.scannet_2d3d import ScanNet2D3DChunks
from mvpnet.evaluate_3d import Evaluator


def ensemble(run_name, split='test'):
    output_dir = '/home/docker_user/workspace/mvpnet_private/outputs/scannet/'
    submit_dir0 = osp.join(output_dir, 'mvpnet_3d_pn2ssg_unet_resnet34_v2_160x120_3views_3nn_adam_log_weights_use_2d_log_weights_training_cotrain/submit/12-17_08-35-45.rits-computervision-salsa_5views/logits/')
    submit_dir1 = osp.join(output_dir, 'mvpnet_3d_pn2ssg_unet_resnet34_v2_160x120_3views_3nn_adam_log_weights_use_2d_log_weights_training/submit/12-16_22-04-41.rits-computervision-salsa_5views/logits/')
    submit_dir2 = osp.join(output_dir, 'mvpnet_3d_pn2ssg_unet_resnet34_v2_160x120_5views_3nn_adam_log_weights_use_2d_log_weights_training/submit/12-18_17-13-56.rits-computervision-salsa/logits/')
    submit_dir3 = osp.join(output_dir, 'mvpnet_3d_pn2ssg_unet_resnet34_v2_160x120_5views_3nn_adam/submit/12-18_17-14-05.rits-computervision-salsa/logits/')
    ensemble_save_dir = osp.join(output_dir, 'ensemble', run_name)
    os.makedirs(ensemble_save_dir)

    dataset = None
    if split == 'val':
        dataset = ScanNet2D3DChunks('/home/docker_user/workspace/mvpnet_private/data/ScanNet/cache_rgbd', '', 'val')
        data = sorted(dataset.data, key=lambda k: k['scan_id'])
        evaluator = Evaluator(dataset.class_names)

    logit_fnames0 = sorted(os.listdir(submit_dir0))
    logit_fnames1 = sorted(os.listdir(submit_dir1))

    assert logit_fnames0 == logit_fnames1
    for i, fname in enumerate(logit_fnames0):
        scan_id, _ = osp.splitext(fname)
        print('{}/{}: {}'.format(i + 1, len(logit_fnames0), scan_id))
        pred_logits_whole_scene0 = np.load(osp.join(submit_dir0, fname))
        pred_logits_whole_scene1 = np.load(osp.join(submit_dir1, fname))
        pred_logits_whole_scene2 = np.load(osp.join(submit_dir2, fname))
        pred_logits_whole_scene3 = np.load(osp.join(submit_dir3, fname))
        pred_logits_whole_scene = scipy.special.softmax(pred_logits_whole_scene0, axis=1) + \
                                  scipy.special.softmax(pred_logits_whole_scene1, axis=1) + \
                                  scipy.special.softmax(pred_logits_whole_scene2, axis=1) + \
                                  scipy.special.softmax(pred_logits_whole_scene3, axis=1)
        pred_labels_whole_scene = pred_logits_whole_scene.argmax(1)

        if dataset is not None:
            seg_label = data[i]['seg_label']
            seg_label = dataset.nyu40_to_scannet[seg_label]
            evaluator.update(pred_labels_whole_scene, seg_label)

        # save to txt file for submission
        remapped_pred_labels = scannet_to_nyu40[pred_labels_whole_scene]
        np.savetxt(osp.join(ensemble_save_dir, scan_id + '.txt'), remapped_pred_labels, '%d')

    if dataset is not None:
        print('overall accuracy={:.2f}%'.format(100.0 * evaluator.overall_acc))
        print('overall IOU={:.2f}'.format(100.0 * evaluator.overall_iou))
        print('class-wise accuracy and IoU.\n{}'.format(evaluator.print_table()))
        evaluator.save_table(osp.join(ensemble_save_dir, 'eval.{}.tsv'.format(run_name)))


if __name__ == '__main__':
    # run name
    timestamp = time.strftime('%m-%d_%H-%M-%S')
    hostname = socket.gethostname()
    run_name = '{:s}.{:s}'.format(timestamp, hostname)
    ensemble(run_name, 'val')
