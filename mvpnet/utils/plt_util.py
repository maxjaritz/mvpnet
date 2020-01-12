"""Matplotlib visualization helpers."""
import numpy as np
from matplotlib import pyplot as plt


def imshows(images, titles=None, suptitle=None, filename=None):
    """Show multiple images"""
    fig = plt.figure(figsize=[len(images) * 8, 8])
    for ind, image in enumerate(images):
        ax = fig.add_subplot(1, len(images), ind + 1)
        ax.imshow(image)
        if titles is not None:
            ax.set_title(titles[ind])
        ax.set_axis_off()
    plt.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9, wspace=0.01, hspace=0.01)
    if suptitle:
        plt.suptitle(suptitle)
    if filename:
        fig.savefig(filename)
    else:
        plt.show()
    plt.close(fig)
