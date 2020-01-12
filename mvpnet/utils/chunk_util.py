import numpy as np


def scene2chunks_legacy(points, chunk_size, stride, thresh=1000, margin=(0.2, 0.2), return_bbox=False):
    """Split the whole scene into chunks based on the original PointNet++ implementation.
    Only slide chunks on the xy-plane

    Args:
        points (np.ndarray): (num_points, 3)
        chunk_size (2-tuple): size of chunk
        stride (float): stride of chunk
        thresh (int): minimum number of points in a qualified chunk
        margin (2-tuple): margin of chunk
        return_bbox (bool): whether to return bounding boxes

    Returns:
        chunk_indices (list of np.ndarray)
        chunk_bboxes (list of np.ndarray, optional): each bbox is (x1, y1, z1, x2, y2, z2)

    """
    chunk_size = np.asarray(chunk_size)
    margin = np.asarray(margin)

    coord_max = np.max(points, axis=0)  # max x,y,z
    coord_min = np.min(points, axis=0)  # min x,y,z
    limit = coord_max - coord_min
    # get the corner of chunks.
    num_chunks = np.ceil((limit[:2] - chunk_size) / stride).astype(int) + 1
    corner_list = []
    for i in range(num_chunks[0]):
        for j in range(num_chunks[1]):
            corner_list.append((coord_min[0] + i * stride, coord_min[1] + j * stride))

    xy = points[:, :2]
    chunk_indices = []
    chunk_bboxes = []
    for corner in corner_list:
        corner = np.asarray(corner)
        mask = np.all(np.logical_and(xy >= corner, xy <= corner + chunk_size), axis=1)
        # discard unqualified chunks
        if np.sum(mask) < thresh:
            continue
        mask = np.all(np.logical_and(xy >= corner - margin, xy <= corner + chunk_size + margin), axis=1)
        indices = np.nonzero(mask)[0]
        chunk_indices.append(indices)
        if return_bbox:
            chunk = points[indices]
            bbox = np.hstack([corner - margin, chunk.min(0)[2], corner + chunk_size + margin, chunk.max(0)[2]])
            chunk_bboxes.append(bbox)
    if return_bbox:
        return chunk_indices, chunk_bboxes
    else:
        return chunk_indices
