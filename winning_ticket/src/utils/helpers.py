import functools
import torch


def get_layers(model):
    conv_mods = []
    linear_mods = []
    bn_mods = []
    for n, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            conv_mods.append((n, module))
        elif isinstance(module, torch.nn.Linear):
            linear_mods.append((n, module))
        elif isinstance(module, torch.nn.BatchNorm2d):
            bn_mods.append((n, module))
    return conv_mods, linear_mods, bn_mods


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))