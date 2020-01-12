"""Helpers for operating modules/parameters

Notes:
    Useful regex expression
    1. nothing else classifier: '^((?!classifier).)*$'

"""

import re
import logging

import torch.nn as nn


class Freezer(object):
    def __init__(self, module, patterns):
        self.module = module
        self.patterns = patterns

    def freeze(self, verbose=False, logger=None):
        freeze_by_patterns(self.module, self.patterns)
        if verbose:
            frozen_modules = [name for name, m in self.module.named_modules() if not m.training]
            frozen_params = [name for name, params in self.module.named_parameters() if not params.requires_grad]
            _print = print if logger is None else logging.info
            for name in frozen_modules:
                _print('Module {} is frozen.'.format(name))
            for name in frozen_params:
                _print('Params {} is frozen.'.format(name))


def apply_params(module, patterns, requires_grad=False):
    """Apply freeze/unfreeze on parameters

    Args:
        module (torch.nn.Module): the module to apply
        patterns (sequence of str): strings which define all the patterns of interests
        requires_grad (bool, optional): whether to freeze params

    """
    for name, params in module.named_parameters():
        for pattern in patterns:
            assert isinstance(pattern, str)
            if re.search(pattern, name):
                params.requires_grad = requires_grad


def apply_modules(module, patterns, mode=False, prefix=''):
    """Apply train/eval on modules

    Args:
        module (torch.nn.Module): the module to apply
        patterns (sequence of str): strings which define all the patterns of interests
        mode (bool, optional): whether to set the module training mode
        prefix (str, optional)

    """
    for name, m in module._modules.items():
        for pattern in patterns:
            assert isinstance(pattern, str)
            full_name = prefix + ('.' if prefix else '') + name
            if re.search(pattern, full_name):
                # avoid redundant call
                m.train(mode)
            else:
                apply_modules(m, patterns, mode=mode, prefix=full_name)


def freeze_by_patterns(module, patterns):
    """Freeze by matching patterns"""
    param_list = []
    module_list = []
    for pattern in patterns:
        if pattern.startswith('module:'):
            module_list.append(pattern[7:])
        else:
            param_list.append(pattern)
    apply_params(module, param_list, requires_grad=False)
    apply_modules(module, module_list, mode=False)


def unfreeze_by_patterns(module, patterns):
    """Unfreeze module by matching patterns"""
    param_list = []
    module_list = []
    for pattern in patterns:
        if pattern.startswith('module:'):
            module_list.append(pattern[7:])
        else:
            param_list.append(pattern)
    apply_params(module, param_list, requires_grad=True)
    apply_modules(module, module_list, mode=True)


def apply_bn(module, mode, requires_grad):
    """Modify batch normalization in the module

    Args:
        module (nn.Module): the module to operate
        mode (bool): train/eval mode
        requires_grad (bool): whether parameters require gradients

    Notes:
        Note that the difference between the behaviors of BatchNorm.eval() and BatchNorm(track_running_stats=False)

    """
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.train(mode)
            for params in m.parameters():
                params.requires_grad = requires_grad


def freeze_bn(module):
    apply_bn(module, mode=False, requires_grad=False)
