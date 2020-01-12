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
from sklearn.neighbors import NearestNeighbors


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
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--nb-pts', type=int, default=32768, help='number of points')
    parser.add_argument('--num-votes', type=int, default=3, help='number of votes')
    parser.add_argument('--no-rot', action='store_true', help='disable rotation augmentation')
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
    aug_list = []
    if not args.no_rot:
        aug_list.append(T.RandomRotateZ())
    transform = T.Compose([T.ToTensor(), T.Transpose()])
    use_color = args.use_color or (model.in_channels == 3)
    num_votes = args.num_votes

    # ---------------------------------------------------------------------------- #
    # Test
    # ---------------------------------------------------------------------------- #
    model.eval()
    set_random_seed(args.seed)
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

            # prepare inputs
            tic = time.time()
            data_list = []
            points_list = []
            colors_list = []
            ind_list = []
            for vote_ind in range(num_votes):
                if len(points) >= args.nb_pts:
                    ind = np.random.choice(len(points), size=args.nb_pts, replace=False)
                else:
                    ind = np.hstack([np.arange(len(points)), np.zeros(args.nb_pts - len(points))])
                points_list.append(points[ind])
                colors_list.append(colors[ind])
                ind_list.append(ind)
            for vote_ind in range(num_votes):
                data_single = {
                    'points': points_list[vote_ind],
                }
                if use_color:
                    data_single['feature'] = colors_list[vote_ind]
                data_list.append(transform(**data_single))
            data_batch = {k: torch.stack([x[k] for x in data_list]) for k in data_single}
            data_batch = {k: v.cuda(non_blocking=True) for k, v in data_batch.items()}
            preprocess_time = time.time() - tic

            # forward
            tic = time.time()
            preds = model(data_batch)
            seg_logit_batch = preds['seg_logit'].cpu().numpy()
            forward_time = time.time() - tic

            # propagate predictions and ensemble
            tic = time.time()
            pred_logit_whole_scene = np.zeros([len(points), num_classes], dtype=np.float32)
            for vote_ind in range(num_votes):
                points_per_vote = points_list[vote_ind]
                seg_logit_per_vote = seg_logit_batch[vote_ind].T
                # Propagate to nearest neighbours
                nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(points_per_vote)
                _, nn_indices = nbrs.kneighbors(points[:, 0:3])
                pred_logit_whole_scene += seg_logit_per_vote[nn_indices[:, 0]]
            # if we use softmax, it is necessary to normalize logits
            pred_logit_whole_scene = pred_logit_whole_scene / num_votes
            pred_label_whole_scene = np.argmax(pred_logit_whole_scene, axis=1)
            postprocess_time = time.time() - tic

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
            test_meters.update(data=data_time,
                               preprocess_time=preprocess_time,
                               forward_time=forward_time,
                               postprocess_time=postprocess_time,
                               metric_time=metric_time)

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
