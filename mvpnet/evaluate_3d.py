import numpy as np
from sklearn.metrics import confusion_matrix as CM

CLASS_NAMES = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
               'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain',
               'refridgerator', 'showercurtain', 'toilet', 'sink', 'bathtub', 'otherfurniture',
               ]
EVAL_CLASS_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]


class Evaluator(object):
    def __init__(self, class_names, labels=None):
        self.class_names = tuple(class_names)
        self.num_classes = len(class_names)
        self.labels = np.arange(self.num_classes) if labels is None else np.array(labels)
        assert self.labels.shape[0] == self.num_classes
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def update(self, pred_label, gt_label):
        """Update per instance

        Args:
            pred_label (np.ndarray): (num_points)
            gt_label (np.ndarray): (num_points,)

        """
        # convert ignore_label to num_classes
        # refer to sklearn.metrics.confusion_matrix
        if np.all(gt_label < 0):
            print('Invalid label.')
            return
        gt_label[gt_label == -100] = self.num_classes
        confusion_matrix = CM(gt_label.flatten(),
                              pred_label.flatten(),
                              labels=self.labels)
        self.confusion_matrix += confusion_matrix

    def batch_update(self, pred_labels, gt_labels):
        assert len(pred_labels) == len(gt_labels)
        for pred_label, gt_label in zip(pred_labels, gt_labels):
            self.update(pred_label, gt_label)

    @property
    def overall_acc(self):
        return np.sum(np.diag(self.confusion_matrix)) / np.sum(self.confusion_matrix)

    @property
    def overall_iou(self):
        return np.nanmean(self.class_iou)

    @property
    def class_seg_acc(self):
        return [self.confusion_matrix[i, i] / np.sum(self.confusion_matrix[i])
                for i in range(self.num_classes)]

    @property
    def class_iou(self):
        iou_list = []
        for i in range(self.num_classes):
            tp = self.confusion_matrix[i, i]
            p = self.confusion_matrix[:, i].sum()
            g = self.confusion_matrix[i, :].sum()
            union = p + g - tp
            if union == 0:
                iou = float('nan')
            else:
                iou = tp / union
            iou_list.append(iou)
        return iou_list

    def print_table(self):
        from tabulate import tabulate
        header = ['Class', 'Accuracy', 'IOU', 'Total']
        seg_acc_per_class = self.class_seg_acc
        iou_per_class = self.class_iou
        table = []
        for ind, class_name in enumerate(self.class_names):
            table.append([class_name,
                          seg_acc_per_class[ind] * 100,
                          iou_per_class[ind] * 100,
                          int(self.confusion_matrix[ind].sum()),
                          ])
        return tabulate(table, headers=header, tablefmt='psql', floatfmt='.2f')

    def save_table(self, filename):
        from tabulate import tabulate
        header = ('overall acc', 'overall iou') + self.class_names
        table = [[self.overall_acc, self.overall_iou] + self.class_iou]
        with open(filename, 'w') as f:
            # In order to unify format, remove all the alignments.
            f.write(tabulate(table, headers=header, tablefmt='tsv', floatfmt='.5f',
                             numalign=None, stralign=None))


def main():
    """Integrated official evaluation scripts
    Use multiple threads to process in parallel

    References: https://github.com/ScanNet/ScanNet/blob/master/BenchmarkScripts/3d_evaluation/evaluate_semantic_label.py
    """
    import os
    import sys
    import argparse
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(description='Evaluate mIoU on ScanNetV2')
    parser.add_argument(
        '--pred-path', type=str, help='path to prediction',
    )
    parser.add_argument(
        '--gt-path', type=str, help='path to ground-truth',
    )
    args = parser.parse_args()

    pred_files = [f for f in os.listdir(args.pred_path) if f.endswith('.txt')]
    gt_files = []
    if len(pred_files) == 0:
        raise RuntimeError('No result files found.')
    for i in range(len(pred_files)):
        gt_file = os.path.join(args.gt_path, pred_files[i])
        if not os.path.isfile(gt_file):
            raise RuntimeError('Result file {} does not match any gt file'.format(pred_files[i]))
        gt_files.append(gt_file)
        pred_files[i] = os.path.join(args.pred_path, pred_files[i])

    evaluator = Evaluator(CLASS_NAMES, EVAL_CLASS_IDS)
    print('evaluating', len(pred_files), 'scans...')

    dataloader = DataLoader(list(zip(pred_files, gt_files)), batch_size=1, num_workers=4,
                            collate_fn=lambda x: tuple(np.loadtxt(xx, dtype=np.uint8) for xx in x[0]))

    # sync
    # for i in range(len(pred_files)):
    #     # It takes a long time to load data.
    #     pred_label = np.loadtxt(pred_files[i], dtype=np.uint8)
    #     gt_label = np.loadtxt(gt_files[i], dtype=np.uint8)
    #     evaluator.update(pred_label, gt_label)
    #     sys.stdout.write("\rscans processed: {}".format(i + 1))
    #     sys.stdout.flush()

    # async, much faster
    for i, (pred_label, gt_label) in enumerate(dataloader):
        evaluator.update(pred_label, gt_label)
        sys.stdout.write("\rscans processed: {}".format(i + 1))
        sys.stdout.flush()

    print('')
    print(evaluator.print_table())


if __name__ == '__main__':
    main()
