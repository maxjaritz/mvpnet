from torch import nn
import torch.nn.functional as F

from .conv import Conv1dBNReLU, Conv2dBNReLU
from .linear import LinearBNReLU


class MLP(nn.ModuleList):
    def __init__(self,
                 in_channels,
                 mlp_channels,
                 bn=True):
        """Multi-layer perception with relu activation

        Args:
            in_channels (int): the number of channels of input tensor
            mlp_channels (tuple): the numbers of channels of fully connected layers
            bn (bool): whether to use batch normalization

        """
        super(MLP, self).__init__()

        self.in_channels = in_channels
        self.out_channels = mlp_channels[-1]

        c_in = in_channels
        for ind, c_out in enumerate(mlp_channels):
            self.append(LinearBNReLU(c_in, c_out, relu=True, bn=bn))
            c_in = c_out

    def forward(self, x):
        for module in self:
            assert isinstance(module, LinearBNReLU)
            x = module(x)
        return x


class SharedMLP(nn.ModuleList):
    def __init__(self,
                 in_channels,
                 mlp_channels,
                 ndim=1,
                 bn=True):
        """Multi-layer perception shared on resolution (1D or 2D)

        Args:
            in_channels (int): the number of channels of input tensor
            mlp_channels (tuple): the numbers of channels of fully connected layers
            ndim (int): the number of dimensions to share
            bn (bool): whether to use batch normalization

        """
        super(SharedMLP, self).__init__()

        self.in_channels = in_channels
        self.out_channels = mlp_channels[-1]
        self.ndim = ndim

        if ndim == 1:
            mlp_module = Conv1dBNReLU
        elif ndim == 2:
            mlp_module = Conv2dBNReLU
        else:
            raise ValueError('SharedMLP only supports ndim=(1, 2).')

        c_in = in_channels
        for ind, c_out in enumerate(mlp_channels):
            self.append(mlp_module(c_in, c_out, 1, relu=True, bn=bn))
            c_in = c_out

    def forward(self, x):
        for module in self:
            assert isinstance(module, (Conv1dBNReLU, Conv2dBNReLU))
            x = module(x)
        return x


class SharedMLPDO(SharedMLP):
    """Shared MLP with dropout"""

    def __init__(self, *args, p=0.5, **kwargs):
        super(SharedMLPDO, self).__init__(*args, **kwargs)
        self.p = p
        self.dropout_fn = F.dropout if self.ndim == 1 else F.dropout2d

    def forward(self, x):
        for module in self:
            assert isinstance(module, (Conv1dBNReLU, Conv2dBNReLU))
            x = module(x)
            # Note that inplace does not work.
            x = self.dropout_fn(x, p=self.p, training=self.training, inplace=False)
        return x

    def extra_repr(self):
        return 'p={}'.format(self.p)
