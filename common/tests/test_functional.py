import numpy as np
from scipy.special import softmax
import scipy.spatial.distance as sdist
import torch

from common.nn.functional import bpdist, bpdist2, pdist2
from common.nn.functional import encode_one_hot, smooth_cross_entropy
from common.nn.functional import batch_index_select


def test_bpdist():
    batch_size = 16
    channels = 64
    num_inst = 1024

    feature_np = np.random.rand(batch_size, channels, num_inst)
    feature_tensor = torch.from_numpy(feature_np)
    if torch.cuda.is_available():
        feature_tensor = feature_tensor.cuda()

    # check pairwise distance
    distance_np = np.stack([sdist.squareform(np.square(sdist.pdist(x.T))) for x in feature_np])
    distance_tensor = bpdist(feature_tensor)
    np.testing.assert_allclose(distance_np, distance_tensor.cpu().numpy(), atol=1e-6)

    # with torch.autograd.profiler.profile(use_cuda=torch.cuda.is_available()) as prof:
    #     bpdist(feature_tensor)
    # print(prof)
    # print(torch.cuda.max_memory_allocated() / (1024.0 ** 2))


def test_bpdist2():
    batch_size = 16
    channels = 64
    num_inst1 = 1023
    num_inst2 = 1025

    feature1_np = np.random.rand(batch_size, channels, num_inst1)
    feature2_np = np.random.rand(batch_size, channels, num_inst2)
    feature1_tensor = torch.from_numpy(feature1_np)
    feature2_tensor = torch.from_numpy(feature2_np)
    if torch.cuda.is_available():
        feature1_tensor = feature1_tensor.cuda()
        feature2_tensor = feature2_tensor.cuda()

    # check pairwise distance_np
    distance_np = np.stack([np.square(sdist.cdist(x.T, y.T)) for x, y in zip(feature1_np, feature2_np)])
    distance_tensor = bpdist2(feature1_tensor, feature2_tensor)  # warm up
    np.testing.assert_allclose(distance_np, distance_tensor.cpu().numpy())

    # with torch.autograd.profiler.profile(use_cuda=torch.cuda.is_available()) as prof:
    #     bpdist2(feature1_tensor, feature2_tensor)
    # print(prof)


def test_pdist2():
    channels = 64
    num_inst1 = 1023
    num_inst2 = 1025

    feature1_np = np.random.rand(num_inst1, channels)
    feature2_np = np.random.rand(num_inst2, channels)
    feature1_tensor = torch.from_numpy(feature1_np)
    feature2_tensor = torch.from_numpy(feature2_np)
    if torch.cuda.is_available():
        feature1_tensor = feature1_tensor.cuda()
        feature2_tensor = feature2_tensor.cuda()

    # check pairwise distance
    distance_np = np.square(sdist.cdist(feature1_np, feature2_np))
    distance_tensor = pdist2(feature1_tensor, feature2_tensor)  # warm up
    np.testing.assert_allclose(distance_np, distance_tensor.cpu().numpy())

    # with torch.autograd.profiler.profile(use_cuda=torch.cuda.is_available()) as prof:
    #     pdist2(feature1_tensor, feature2_tensor)
    # print(prof)


def test_smooth_cross_entropy():
    num_samples = 2
    num_classes = 10
    label_smoothing = 0.1

    # numpy version
    target_np = np.random.randint(0, num_classes, [num_samples])
    one_hot_np = np.zeros([num_samples, num_classes])
    one_hot_np[np.arange(num_samples), target_np] = 1.0
    smooth_one_hot = one_hot_np * (1.0 - label_smoothing) + np.ones_like(one_hot_np) * label_smoothing / num_classes
    logit_np = np.random.randn(num_samples, num_classes)
    prob_np = softmax(logit_np, axis=-1)
    cross_entropy_np = - (smooth_one_hot * np.log(prob_np)).sum(1).mean()

    target = torch.from_numpy(target_np)
    logit = torch.from_numpy(logit_np)

    one_hot = encode_one_hot(target, num_classes)
    np.testing.assert_allclose(one_hot_np, one_hot.numpy())

    cross_entropy = smooth_cross_entropy(logit, target, label_smoothing)
    np.testing.assert_allclose(cross_entropy_np, cross_entropy.numpy())


def test_batch_index_select():
    shape = (2, 16, 9, 32)
    batch_size = shape[0]
    input_np = np.random.randn(*shape)

    for dim in range(1, len(shape)):
        num_select = np.random.randint(shape[dim])
        index_np = np.random.randint(shape[dim], size=(batch_size, num_select))
        target_np = np.stack([np.take(input_np[b], index_np[b], axis=dim - 1) for b in range(batch_size)], axis=0)

        input_tensor = torch.tensor(input_np)
        index_tensor = torch.tensor(index_np)
        target_tensor = batch_index_select(input_tensor, index_tensor, dim=dim)
        np.testing.assert_allclose(target_np, target_tensor.numpy())
