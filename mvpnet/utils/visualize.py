import numpy as np
import open3d as o3d

# color palette for nyu40 labels
# Reference: https://github.com/ScanNet/ScanNet/blob/master/BenchmarkScripts/util.py
NYU40_COLOR_PALETTE = [
    (0, 0, 0),
    (174, 199, 232),  # wall
    (152, 223, 138),  # floor
    (31, 119, 180),  # cabinet
    (255, 187, 120),  # bed
    (188, 189, 34),  # chair
    (140, 86, 75),  # sofa
    (255, 152, 150),  # table
    (214, 39, 40),  # door
    (197, 176, 213),  # window
    (148, 103, 189),  # bookshelf
    (196, 156, 148),  # picture
    (23, 190, 207),  # counter
    (178, 76, 76),
    (247, 182, 210),  # desk
    (66, 188, 102),
    (219, 219, 141),  # curtain
    (140, 57, 197),
    (202, 185, 52),
    (51, 176, 203),
    (200, 54, 131),
    (92, 193, 61),
    (78, 71, 183),
    (172, 114, 82),
    (255, 127, 14),  # refrigerator
    (91, 163, 138),
    (153, 98, 156),
    (140, 153, 101),
    (158, 218, 229),  # shower curtain
    (100, 125, 154),
    (178, 127, 135),
    (120, 185, 128),
    (146, 111, 194),
    (44, 160, 44),  # toilet
    (112, 128, 144),  # sink
    (96, 207, 209),
    (227, 119, 194),  # bathtub
    (213, 92, 176),
    (94, 106, 211),
    (82, 84, 163),  # otherfurn
    (100, 85, 144)
]

SCANNET_COLOR_PALETTE = [
    (174, 199, 232),  # wall
    (152, 223, 138),  # floor
    (31, 119, 180),  # cabinet
    (255, 187, 120),  # bed
    (188, 189, 34),  # chair
    (140, 86, 75),  # sofa
    (255, 152, 150),  # table
    (214, 39, 40),  # door
    (197, 176, 213),  # window
    (148, 103, 189),  # bookshelf
    (196, 156, 148),  # picture
    (23, 190, 207),  # counter
    (247, 182, 210),  # desk
    (219, 219, 141),  # curtain
    (255, 127, 14),  # refrigerator
    (158, 218, 229),  # shower curtain
    (44, 160, 44),  # toilet
    (112, 128, 144),  # sink
    (227, 119, 194),  # bathtub
    (82, 84, 163),  # otherfurn
]


def label2color(labels, colors=None, style='scannet'):
    assert isinstance(labels, np.ndarray) and labels.ndim == 1
    if style == 'scannet':
        color_palette = np.array(SCANNET_COLOR_PALETTE) / 255.
    elif style == 'nyu40_raw':
        color_palette = np.array(NYU40_COLOR_PALETTE) / 255.
    elif style == 'nyu40':
        color_palette = np.array(NYU40_COLOR_PALETTE[1:]) / 255.
    else:
        raise KeyError('Unknown style: {}'.format(style))
    if colors is None:
        colors = np.zeros([labels.shape[0], 3])
    else:
        assert colors.ndim == 2 and colors.shape[1] == 3
        colors = colors.copy()
    mask = (labels >= 0)
    colors[mask] = color_palette[labels[mask]]
    return colors


# ---------------------------------------------------------------------------- #
# Visualize by labels
# ---------------------------------------------------------------------------- #
def visualize_labels(points, seg_label, colors=None, style='scannet'):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points[:, :3])
    pc.colors = o3d.utility.Vector3dVector(label2color(seg_label, colors, style=style))
    geometries = [pc]
    geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0]))
    o3d.visualization.draw_geometries(geometries)
