import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


# Not learning weights, finding subnet
class SplitConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        self.split_mode = kwargs.pop('split_mode', None)
        self.split_rate = kwargs.pop('split_rate', None)
        self.in_channels_order = kwargs.pop('in_channels_order', None)
        # self.keep_rate = keep_rate
        super().__init__(*args, **kwargs)

        if self.split_mode == 'kels':
            if self.in_channels_order is None:
                mask = np.zeros((self.weight.size()))
                if self.weight.size()[1] == 3:  ## This is the first conv
                    mask[:math.ceil(self.weight.size()[0] *
                                    self.split_rate), :, :, :] = 1
                    self.mask_dims = (math.ceil(self.weight.size()[0] *
                                                self.split_rate),
                                      self.weight.size()[1])
                else:
                    mask[:math.ceil(self.weight.size()[0] * self.split_rate
                                    ), :math.ceil(self.weight.size()[1] *
                                                  self.split_rate), :, :] = 1
                    self.mask_dims = (math.ceil(self.weight.size()[0] *
                                                self.split_rate),
                                      math.ceil(self.weight.size()[1] *
                                                self.split_rate))
            else:

                mask = np.zeros((self.weight.size()))
                conv_concat = [
                    int(chs) for chs in self.in_channels_order.split(',')
                ]
                # assert sum(conv_concat) == self.weight.size()[1],'In channels {} should be equal to sum(concat) {}'.format(self.weight.size()[1],conv_concat)
                start_ch = 0
                for conv in conv_concat:
                    mask[:math.ceil(self.weight.size()[0] * self.split_rate),
                         start_ch:start_ch +
                         math.ceil(conv * self.split_rate), :, :] = 1
                    start_ch += conv

        elif self.split_mode == 'wels':
            mask = np.random.rand(*list(self.weight.shape))
            # threshold = np.percentile(scores, (1-self.keep_rate)*100)
            threshold = 1 - self.split_rate
            mask[mask < threshold] = 0
            mask[mask >= threshold] = 1
            if self.split_rate != 1:
                assert len(np.unique(
                    mask)) == 2, 'Something is wrong with the mask {}'.format(
                        np.unique(mask))
        else:
            raise NotImplemented('Invalid split_mode {}'.format(
                self.split_mode))

        self.mask = nn.Parameter(torch.Tensor(mask), requires_grad=False)

    def extract_slim(self):
        c_out, c_in, _, _, = self.weight.size()
        d_out = sum(self.mask[:, 0, 0, 0] == 1).item()
        d_in = sum(self.mask[0, :, 0, 0] == 1).item()
        if self.in_channels_order is None:
            if c_in == 3:
                selected_convs = self.weight[:d_out]
                # is_first_conv = False
            else:
                selected_convs = self.weight[:d_out][:, :d_in, :, :]

            # assert selected_convs.shape == self.mask.shape
            self.weight.data = selected_convs
        else:
            selected_convs = self.weight[:d_out, self.mask[0, :, 0,
                                                           0] == 1, :, :]
            # assert selected_convs.shape == self.mask.shape
            self.weight.data = selected_convs

    def split_reinitialize(self, evolve_mode, device):
        if evolve_mode == 'rand':
            rand_tensor = torch.zeros_like(self.weight).to(device)
            nn.init.kaiming_uniform_(rand_tensor, a=math.sqrt(5))
            self.weight.data = torch.where(self.mask.type(torch.bool),
                                           self.weight.data, rand_tensor)
        else:
            raise NotImplemented('Invalid KE mode {}'.format(evolve_mode))

        if hasattr(self, "bias"
                   ) and self.bias is not None and self.bias_split_rate < 1.0:
            bias_mask = self.mask[:, 0, 0,
                                  0]  ## Same conv mask is used for bias terms
            if evolve_mode == 'rand':
                rand_tensor = torch.zeros_like(self.bias)
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(rand_tensor, -bound, bound)
                self.bias.data = torch.where(bias_mask.type(torch.bool),
                                             self.bias.data, rand_tensor)
            else:
                raise NotImplemented('Invalid KE mode {}'.format(evolve_mode))

    def forward(self, x):
        ## Debugging reasons only
        # if self.split_rate < 1:
        #     w = self.mask * self.weight
        #     if self.bias_split_rate < 1:
        #         # bias_subnet = GetSubnet.apply(self.clamped_bias_scores, self.bias_keep_rate)
        #         b = self.bias * self.mask[:, 0, 0, 0]
        #     else:
        #         b = self.bias
        # else:
        #     w = self.weight
        #     b = self.bias

        w = self.weight
        b = self.bias
        x = F.conv2d(x, w, b, self.stride, self.padding, self.dilation,
                     self.groups)
        return x