from torch import nn


class LinearBNReLU(nn.Module):
    """Applies a linear transformation to the incoming data
    optionally followed by batch normalization and relu activation
    """

    def __init__(self, in_channels, out_channels,
                 relu=True, bn=True):
        super(LinearBNReLU, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.fc = nn.Linear(in_channels, out_channels, bias=(not bn))
        self.bn = nn.BatchNorm1d(out_channels) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.fc(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
