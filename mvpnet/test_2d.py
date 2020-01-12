#!/usr/bin/env python
"""Test 2D semantic segmentation"""

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
from torch.utils.data.dataloader import DataLoader

# Assume that the script is run at the root directory
_ROOT_DIR = os.path.abspath(osp.dirname(__file__) + '/..')
sys.path.insert(0, _ROOT_DIR)
_DEBUG = False

from common.utils.checkpoint import CheckpointerV2
from common.utils.logger import setup_logger
from common.utils.metric_logger import MetricLogger
from common.utils.torch_util import set_random_seed

from mvpnet.models.build import build_model_sem_seg_2d
from mvpnet.data.scannet_2d import ScanNet2D
from mvpnet.evaluate_3d import Evaluator


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
    parser.add_argument('-b', '--batch-size', type=int, help='batch size')
    parser.add_argument('--num-workers', type=int, help='save predictions')
    parser.add_argument('--log-period', type=int, default=100, help='save predictions')
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
    model = build_model_sem_seg_2d(cfg)[0]
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
    test_dataset = ScanNet2D(cfg.DATASET.ROOT_DIR, split=args.split,
                             subsample=None, to_tensor=True,
                             resize=cfg.DATASET.ScanNet2D.resize,
                             normalizer=cfg.DATASET.ScanNet2D.normalizer,
                             )
    batch_size = args.batch_size or cfg.VAL.BATCH_SIZE
    num_workers = args.num_workers or cfg.DATALOADER.NUM_WORKERS
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 drop_last=False)

    # evaluator
    class_names = test_dataset.class_names
    evaluator = Evaluator(class_names)
    num_classes = len(class_names)
    submit_dir = None
    if args.save:
        submit_dir = osp.join(output_dir, 'submit', run_name)

    # ---------------------------------------------------------------------------- #
    # Test
    # ---------------------------------------------------------------------------- #
    model.eval()
    set_random_seed(cfg.RNG_SEED)
    test_meters = MetricLogger(delimiter='  ')

    with torch.no_grad():
        start_time = time.time()
        for iteration, data_batch in enumerate(test_dataloader):
            gt_label = data_batch.get('seg_label', None)
            data_batch = {k: v.cuda(non_blocking=True) for k, v in data_batch.items()}
            # forward
            preds = model(data_batch)
            pred_label = preds['seg_logit'].argmax(1).cpu().numpy()  # (b, h, w)
            # evaluate
            if gt_label is not None:
                gt_label = gt_label.cpu().numpy()
                evaluator.batch_update(pred_label, gt_label)
            # logging
            if args.log_period and iteration % args.log_period == 0:
                logger.info(
                    test_meters.delimiter.join(
                        [
                            '{:d}/{:d}',
                            'acc: {acc:.2f}',
                            'IoU: {iou:.2f}',
                            # '{meters}',
                        ]
                    ).format(
                        iteration, len(test_dataloader),
                        acc=evaluator.overall_acc * 100.0,
                        iou=evaluator.overall_iou * 100.0,
                        # meters=str(test_meters),
                    )
                )
        test_time = time.time() - start_time
        # logger.info('Test {}  test time: {:.2f}s'.format(test_meters.summary_str, test_time))

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
    from mvpnet.config.sem_seg_2d import cfg
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

    assert cfg.TASK == 'sem_seg_2d'
    test(cfg, args, output_dir, run_name)


if __name__ == '__main__':
    main()
