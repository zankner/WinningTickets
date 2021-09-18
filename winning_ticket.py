import copy
import torch

from utils import helpers
from utils.layers import conv, linear, batch_norm


def ticketfy(model, split_rate, split_mode="kels"):
    conv_layers, linear_layers, bn_layers = helpers.get_layers(model)

    for n, _ in conv_layers:
        cur_conv = helpers.rgetattr(model, n)
        helpers.rsetattr(
            model, n,
            conv.SplitConv(cur_conv.in_channels,
                           cur_conv.out_channels,
                           kernel_size=cur_conv.kernel_size,
                           stride=cur_conv.stride,
                           padding=cur_conv.padding,
                           dilation=cur_conv.dilation,
                           groups=cur_conv.groups,
                           bias=cur_conv.bias != None,
                           padding_mode=cur_conv.padding_mode,
                           split_rate=split_rate,
                           split_mode=split_mode))

    for i, (n, _) in enumerate(linear_layers):
        cur_linear = helpers.rgetattr(model, n)
        helpers.rsetattr(
            model, n,
            linear.SplitLinear(cur_linear.in_features,
                               cur_linear.out_features,
                               bias=cur_linear.bias != None,
                               split_rate=split_rate,
                               split_mode=split_mode,
                               last_layer=i == len(linear_layers) - 1))

    for n, _ in bn_layers:
        cur_bn = helpers.rgetattr(model, n)
        helpers.rsetattr(
            model, n,
            batch_norm.SplitBatchNorm(
                cur_bn.num_features,
                eps=cur_bn.eps,
                momentum=cur_bn.momentum,
                track_running_stats=cur_bn.track_running_stats,
                split_rate=split_rate))


def regenerate(model, evolve_mode="rand", device="cpu"):
    for _, m in model.named_modules():
        if hasattr(m, "weight") and m.weight is not None:
            if hasattr(m, "mask"):  ## Conv and Linear but not BN
                assert m.split_rate < 1.0

                if m.__class__ == conv.SplitConv or m.__class__ == linear.SplitLinear:
                    m.split_reinitialize(evolve_mode, device)
                else:
                    raise NotImplemented('Invalid layer {}'.format(
                        m.__class__))


def extract_ticket(model, split_rate):
    split_model = copy.deepcopy(model)
    for n, m in split_model.named_modules():
        if hasattr(m, "weight") and m.weight is not None:
            if hasattr(m, "mask"):
                m.extract_slim()
                # if src_m.__class__ == conv_type.SplitConv:
                # elif src_m.__class__ == linear_type.SplitLinear:
            elif m.__class__ == batch_norm.SplitBatchNorm:  ## BatchNorm has bn_maks not mask
                m.extract_slim()
    return split_model
