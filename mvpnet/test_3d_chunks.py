#!/usr/bin/env python
"""Test semantic segmentation models chunk-by-chunk"""

import os
import os.path as osp
import sys
import argparse
import logging
import time
import socket
import warnings

import numpy as np
import open3d
import torch

# Assume that the script is run at the root directory
_ROOT_DIR = os.path.abspath(osp.dirname(__file__) + '/..')
sys.path.insert(0, _ROOT_DIR)
_DEBUG = False

from common.utils.checkpoint import CheckpointerV2
from common.utils.logger import setup_logger
from common.utils.metric_logger import MetricLogger
from common.utils.torch_util import set_random_seed

from mvpnet.models.build import build_model_sem_seg_3d
from mvpnet.data.scannet_3d import ScanNet3D
from mvpnet.data import transforms as T
from mvpnet.evaluate_3d import Evaluator
from mvpnet.utils.chunk_util import scene2chunks_legacy


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch 3D Deep Learning Test')
    parser.add_argument(
        '--cfg',
        dest='config_file',
        default='',
        metavar='FILE',
        help='path to config file',
        type=str,
    )
    parser.add_argument('--ckpt-path', type=str, help='path to checkpoint file')
    parser.add_argument('--split', type=str, default='val', help='split')
    parser.add_argument('--save', action='store_true', help='save predictions')
    parser.add_argument('--chunk-size', type=float, default=1.5, help='chunk size')
    parser.add_argument('--chunk-stride', type=float, default=0.5, help='chunk stride')
    parser.add_argument('--chunk-thresh', type=int, default=1000, help='chunk threshold')
    parser.add_argument('--min-nb-pts', type=int, default=2048, help='minimum number of points in the chunk')
    parser.add_argument('--use-color', action='store_true', help='whether to use colors')
    parser.add_argument(
        'opts',
        help='Modify config options using the command-line',
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    return args


def test(cfg, args, output_dir='', run_name=''):
    logger = logging.getLogger('mvpnet.test')

    # build model
    model = build_model_sem_seg_3d(cfg)[0]
    model = model.cuda()

    # build checkpointer
    checkpointer = CheckpointerV2(model, save_dir=output_dir, logger=logger)

    if args.ckpt_path:
        # load weight if specified
        weight_path = args.ckpt_path.replace('@', output_dir)
        checkpointer.load(weight_path, resume=False)
    else:
        # load last checkpoint
        checkpointer.load(None, resume=True)

    # build dataset
    test_dataset = ScanNet3D(cfg.DATASET.ROOT_DIR, split=args.split)
    test_dataset.set_mapping('scannet')

    # evaluator
    class_names = test_dataset.class_names
    evaluator = Evaluator(class_names)
    num_classes = len(class_names)
    submit_dir = None
    if args.save:
        submit_dir = osp.join(output_dir, 'submit', run_name)

    # others
    transform = T.Compose([T.ToTensor(), T.Pad(args.min_nb_pts), T.Transpose()])
    use_color = args.use_color or (model.in_channels == 3)

    # ---------------------------------------------------------------------------- #
    # Test
    # ---------------------------------------------------------------------------- #
    model.eval()
    set_random_seed(cfg.RNG_SEED)
    test_meters = MetricLogger(delimiter='  ')

    with torch.no_grad():
        start_time = time.time()
        for scan_idx in range(len(test_dataset)):
            start_time_scan = time.time()
            # fetch data
            tic = time.time()
            data_dict = test_dataset[scan_idx]
            scan_id = data_dict['scan_id']
            points = data_dict['points']  # (n, 3)
            colors = data_dict['colors']  # (n, 3)
            seg_label = data_dict.get('seg_label', None)  # (n,)
            data_time = time.time() - tic

            # generate chunks
            tic = time.time()
            chunk_indices = scene2chunks_legacy(points,
                                                chunk_size=(args.chunk_size, args.chunk_size),
                                                stride=args.chunk_stride,
                                                thresh=args.chunk_thresh)
            # num_chunks = len(chunk_indices)
            preprocess_time = time.time() - tic

            # prepare outputs
            num_points = len(points)
            pred_logit_whole_scene = np.zeros([num_points, num_classes], dtype=np.float32)
            num_pred_per_point = np.zeros(num_points, dtype=np.uint8)

            # iterate over chunks
            tic = time.time()
            for indices in chunk_indices:
                chunk_points = points[indices]
                chunk_feature = colors[indices]
                chunk_num_points = len(chunk_points)
                if chunk_num_points < args.min_nb_pts:
                    print('Too few points({}) in a chunk of {}'.format(chunk_num_points, scan_id))
                # if _DEBUG:
                #     # DEBUG: visualize chunk
                #     from mvpnet.utils.o3d_util import visualize_point_cloud
                #     visualize_point_cloud(chunk_points, colors=chunk_feature)
                # prepare inputs
                data_batch = {'points': chunk_points}
                if use_color:
                    data_batch['feature'] = chunk_feature
                # preprocess
                data_batch = transform(**data_batch)
                data_batch = {k: torch.stack([v]) for k, v in data_batch.items()}
                data_batch = {k: v.cuda(non_blocking=True) for k, v in data_batch.items()}
                # forward
                preds = model(data_batch)
                seg_logit = preds['seg_logit'].squeeze(0).cpu().numpy().T
                seg_logit = seg_logit[:chunk_num_points]
                # update
                pred_logit_whole_scene[indices] += seg_logit
                num_pred_per_point[indices] += 1
            forward_time = time.time() - tic

            pred_logit_whole_scene = pred_logit_whole_scene / np.maximum(num_pred_per_point[:, np.newaxis], 1)
            pred_label_whole_scene = np.argmax(pred_logit_whole_scene, axis=1)

            no_pred_mask = num_pred_per_point == 0
            no_pred_indices = np.nonzero(no_pred_mask)[0]
            if no_pred_indices.size > 0:
                logger.warning('{:s}: There are {:d} points without prediction.'.format(scan_id, no_pred_mask.sum()))
                pred_label_whole_scene[no_pred_indices] = num_classes

            if _DEBUG:
                # DEBUG: visualize scene
                from mvpnet.utils.visualize import visualize_labels
                visualize_labels(points, pred_label_whole_scene, colors=colors)

            # evaluate
            tic = time.time()
            if seg_label is not None:
                evaluator.update(pred_label_whole_scene, seg_label)
            metric_time = time.time() - tic

            batch_time = time.time() - start_time_scan
            test_meters.update(time=batch_time)
            test_meters.update(data=data_time, preprocess_time=preprocess_time,
                               forward_time=forward_time, metric_time=metric_time)

            # save prediction
            if submit_dir:
                remapped_pred_labels = test_dataset.scannet_to_raw[pred_label_whole_scene]
                np.savetxt(osp.join(submit_dir, scan_id + '.txt'), remapped_pred_labels, '%d')

            logger.info(
                test_meters.delimiter.join(
                    [
                        '{:d}/{:d}({:s})',
                        'acc: {acc:.2f}',
                        'IoU: {iou:.2f}',
                        '{meters}',
                    ]
                ).format(
                    scan_idx, len(test_dataset), scan_id,
                    acc=evaluator.overall_acc * 100.0,
                    iou=evaluator.overall_iou * 100.0,
                    meters=str(test_meters),
                )
            )
        test_time = time.time() - start_time
        logger.info('Test {}  test time: {:.2f}s'.format(test_meters.summary_str, test_time))

    # evaluate
    logger.info('overall accuracy={:.2f}%'.format(100.0 * evaluator.overall_acc))
    logger.info('overall IOU={:.2f}'.format(100.0 * evaluator.overall_iou))
    logger.info('class-wise accuracy and IoU.\n{}'.format(evaluator.print_table()))
    evaluator.save_table(osp.join(output_dir, 'eval.{}.tsv'.format(run_name)))


def main():
    args = parse_args()

    # load the configuration
    # import on-the-fly to avoid overwriting cfg
    from common.config import purge_cfg
    from mvpnet.config.sem_seg_3d import cfg
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    purge_cfg(cfg)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    # replace '@' with config path
    if output_dir:
        config_path = osp.splitext(args.config_file)[0]
        output_dir = output_dir.replace('@', config_path.replace('configs', 'outputs'))
        if not osp.isdir(output_dir):
            warnings.warn('Make a new directory: {}'.format(output_dir))
            os.makedirs(output_dir)

    # run name
    timestamp = time.strftime('%m-%d_%H-%M-%S')
    hostname = socket.gethostname()
    run_name = '{:s}.{:s}'.format(timestamp, hostname)

    logger = setup_logger('mvpnet', output_dir, comment='test.{:s}'.format(run_name))
    logger.info('{:d} GPUs available'.format(torch.cuda.device_count()))
    logger.info(args)

    from common.utils.misc import collect_env_info
    logger.info('Collecting env info (might take some time)\n' + collect_env_info())

    logger.info('Loaded configuration file {:s}'.format(args.config_file))
    logger.info('Running with config:\n{}'.format(cfg))

    assert cfg.TASK == 'sem_seg_3d'
    test(cfg, args, output_dir, run_name)


if __name__ == '__main__':
    main()
