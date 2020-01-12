import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR

from common.solver.lr_scheduler import WarmupMultiStepLR, ClipLR


def test_WarmupMultiStepLR():
    target = [0.5, 0.75] + [1.0] * 3 + [0.1] * 3 + [0.01] * 2
    optimizer = torch.optim.SGD([torch.nn.Parameter()], lr=1.0)
    lr_scheduler = WarmupMultiStepLR(optimizer, milestones=[5, 8], gamma=0.1,
                                     warmup_steps=2, warmup_factor=0.5)
    output = []
    for epoch in range(10):
        output.extend(lr_scheduler.get_lr())
        optimizer.step()
        lr_scheduler.step()
    # print(output)
    np.testing.assert_allclose(output, target, atol=1e-6)


def test_ClipLR():
    target = [0.1 ** i for i in range(4)] + [1e-3]
    optimizer = torch.optim.SGD([torch.nn.Parameter()], lr=1.0)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    lr_scheduler = ClipLR(lr_scheduler, min_lr=1e-3)
    output = []
    for epoch in range(5):
        output.extend(lr_scheduler.get_lr())
        optimizer.step()
        lr_scheduler.step()
    np.testing.assert_allclose(output, target, atol=1e-6)
