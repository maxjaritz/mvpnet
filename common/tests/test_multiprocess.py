"""Test multi-process dataloader

Notes:
    1. Numpy random generator in each worker is same, and even does not affect the generator of the main process.
    2. When num_workers > 1, h5py does not work properly.

References:
    1. https://pytorch.org/docs/stable/notes/faq.html#dataloader-workers-random-seed
    2. https://github.com/pytorch/pytorch/issues/3415

"""

import tempfile
import h5py
import random
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

from common.utils.torch_util import worker_init_fn, set_random_seed


class RandomDataset(Dataset):
    def __init__(self, size=16):
        self.size = size

    def __getitem__(self, index):
        return index, random.random(), np.random.rand(1).item(), torch.rand(1).item()

    def __len__(self):
        return self.size


def test_dataloader():
    set_random_seed(0)
    dataset = RandomDataset()

    # ---------------------------------------------------------------------------- #
    # It is expected that every two batches contain same numpy random results.
    # And even for next round it still gets the same results.
    # ---------------------------------------------------------------------------- #
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=lambda x: x,
        num_workers=2,
        # worker_init_fn=worker_init_fn,
    )

    print('Without worker_init_fn')
    for _ in range(2):
        print('-' * 8)
        for x in dataloader:
            print(x)

    # ---------------------------------------------------------------------------- #
    # By initializing the worker, this issue could be solved.
    # ---------------------------------------------------------------------------- #
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=lambda x: x,
        num_workers=2,
        worker_init_fn=worker_init_fn,
    )

    print('With worker_init_fn')
    for _ in range(2):
        print('-' * 8)
        for x in dataloader:
            print(x)


class H5Dataset(Dataset):
    def __init__(self, filename, size):
        self.size = size
        self.filename = filename
        self.h5 = None

    def load(self):
        return h5py.File(self.filename, mode='r')

    def __getitem__(self, index):
        if self.h5 is None:
            self.h5 = self.load()
        data = self.h5['data'][index]
        return index, data

    def __len__(self):
        return self.size


def test_H5Dataset():
    """Read HDF5 in parallel

    There exist some issues of hdf5 handlers. It could be solved by loading hdf5 on-the-fly.
    However, the drawback is that it will load multiple copies into memory for multiple processes.

    """
    set_random_seed(0)
    size = 10

    with tempfile.TemporaryDirectory() as tmpdirname:
        filename = tmpdirname + '/data.h5'
        h5_file = h5py.File(filename, mode='w')
        h5_file.create_dataset('data', data=np.arange(size))
        h5_file.close()
        dataset = H5Dataset(filename, size)

        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda x: x,
            num_workers=2,
        )

        print('-' * 8)
        for x in dataloader:
            print(x)
