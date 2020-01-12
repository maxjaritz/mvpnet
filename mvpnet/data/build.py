from __future__ import division
from torch.utils.data.dataloader import DataLoader
from common.utils.torch_util import worker_init_fn
from common.utils.sampler import RepeatSampler
from mvpnet.data import transforms as T


def build_dataloader(cfg, mode='train'):
    assert mode in ['train', 'val']
    batch_size = cfg[mode.upper()].BATCH_SIZE
    is_train = (mode == 'train')

    if cfg.TASK == 'sem_seg_3d':
        dataset = build_dataset_3d(cfg, mode)
    elif cfg.TASK == 'sem_seg_2d':
        dataset = build_dataset_2d(cfg, mode)
    elif cfg.TASK == 'mvpnet_3d':
        dataset = build_dataset_mvpnet_3d(cfg, mode)
    else:
        raise NotImplementedError('Unsupported task: {}'.format(cfg.TASK))

    if is_train:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=cfg.DATALOADER.DROP_LAST,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            worker_init_fn=worker_init_fn,
        )
    else:
        sampler = RepeatSampler(dataset, repeats=cfg.VAL.REPEATS)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=False,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            worker_init_fn=worker_init_fn,
        )

    return dataloader


def build_dataset_3d(cfg, mode='train'):
    from mvpnet.data.scannet_3d import ScanNet3DChunks, ScanNet3DScene
    split = cfg.DATASET[mode.upper()]
    is_train = (mode == 'train')

    augmentations = cfg.TRAIN.AUGMENTATION if is_train else cfg.VAL.AUGMENTATION
    transform_list = parse_augmentations(augmentations)
    transform_list.append(T.ToTensor())
    transform_list.append(T.Transpose())
    transform = T.Compose(transform_list)

    dataset_kwargs = cfg.DATASET.get(cfg.DATASET.TYPE, dict())
    if cfg.DATASET.TYPE == 'ScanNet3DChunks':
        dataset = ScanNet3DChunks(root_dir=cfg.DATASET.ROOT_DIR,
                                  split=split,
                                  transform=transform,
                                  **dataset_kwargs)
    elif cfg.DATASET.TYPE == 'ScanNet3DScene':
        dataset = ScanNet3DScene(root_dir=cfg.DATASET.ROOT_DIR,
                                 split=split,
                                 transform=transform,
                                 **dataset_kwargs)
    else:
        raise ValueError('Unsupported type of dataset: {}.'.format(cfg.DATASET.TYPE))

    return dataset


def parse_augmentations(augmentations):
    transform_list = []
    for aug in augmentations:
        if isinstance(aug, (list, tuple)):
            method = aug[0]
            args = aug[1:]
        else:
            method = aug
            args = []
        transform_list.append(getattr(T, method)(*args))
    return transform_list


def build_dataset_2d(cfg, mode='train'):
    from mvpnet.data.scannet_2d import ScanNet2D
    split = cfg.DATASET[mode.upper()]
    is_train = (mode == 'train')

    dataset_kwargs = cfg.DATASET.get(cfg.DATASET.TYPE, dict())
    dataset_kwargs = dict(dataset_kwargs)
    if cfg.DATASET.TYPE == 'ScanNet2D':
        augmentation = dataset_kwargs.pop('augmentation')
        augmentation = augmentation if is_train else dict()
        dataset = ScanNet2D(root_dir=cfg.DATASET.ROOT_DIR,
                            split=split,
                            to_tensor=True,
                            subsample=None if is_train else 100,
                            **dataset_kwargs,
                            **augmentation)
    else:
        raise ValueError('Unsupported type of dataset: {}.'.format(cfg.DATASET.TYPE))

    return dataset


def build_dataset_mvpnet_3d(cfg, mode='train'):
    from mvpnet.data.scannet_2d3d import ScanNet2D3DChunks
    split = cfg.DATASET[mode.upper()]
    is_train = (mode == 'train')

    dataset_kwargs = cfg.DATASET.get(cfg.DATASET.TYPE, dict())
    dataset_kwargs = dict(dataset_kwargs)
    if cfg.DATASET.TYPE == 'ScanNet2D3DChunks':
        augmentation = dataset_kwargs.pop('augmentation')
        augmentation = augmentation if is_train else dict()
        dataset = ScanNet2D3DChunks(split=split,
                                    to_tensor=True,
                                    **dataset_kwargs,
                                    **augmentation)
    else:
        raise ValueError('Unsupported type of dataset: {}.'.format(cfg.DATASET.TYPE))

    return dataset
