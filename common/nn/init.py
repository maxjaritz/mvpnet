from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm


def init_bn(module):
    assert isinstance(module, _BatchNorm)
    if module.weight is not None:
        nn.init.ones_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def set_bn(module, momentum=None, eps=None):
    for m in module.modules():
        if isinstance(m, _BatchNorm):
            if momentum is not None:
                m.momentum = momentum
            if eps is not None:
                m.eps = eps


def xavier_uniform(module):
    if module.weight is not None:
        nn.init.xavier_uniform_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def xavier_normal(module):
    if module.weight is not None:
        nn.init.xavier_normal_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def kaiming_uniform(module):
    if module.weight is not None:
        nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def kaiming_normal(module):
    if module.weight is not None:
        nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
    if module.bias is not None:
        nn.init.zeros_(module.bias)
