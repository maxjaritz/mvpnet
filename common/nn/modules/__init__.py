"""Wrappers of built-in modules

Notes:
    1. Built-in modules usually have built-in initializations
    2. Default initialization of BN has been fixed since pytorch v1.2.0
    3. If BN is applied after convolution, bias is unnecessary.

"""

from .conv import Conv1dBNReLU, Conv2dBNReLU
from .linear import LinearBNReLU
from .mlp import MLP, SharedMLP, SharedMLPDO
