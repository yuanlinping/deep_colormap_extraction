import torch
from torch import nn
import torch.nn.functional as F
"""
Components
"""

class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, rate):  # inplanes: input channel; planes: output channel
        super(ASPP_module, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=1, padding=padding, dilation=rate, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)

        return self.relu(x)

class aspp(nn.Module):
    def __init__(self, output_stride=16, depth=256):
        super(aspp, self).__init__()
        if output_stride not in [8,16]:
            raise ValueError('output_stride must be either 8 or 16.')

        self.atrous_rate = [6,12,18]
        if output_stride == 8:
            self.atrous_rate = [2 * rate for rate in self.atrous_rate]

        self.conv_1x1 = ASPP_module(depth, depth,rate=1)
        self.conv_3x3_1 = ASPP_module(depth, depth, rate=self.atrous_rate[0])
        self.conv_3x3_2 = ASPP_module(depth, depth, rate=self.atrous_rate[1])
        self.conv_3x3_3 = ASPP_module(depth, depth, rate=self.atrous_rate[2])

        self.relu = nn.ReLU()

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(depth, depth, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(depth),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(depth * 5, depth, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(depth)

    def forward(self, x):
        x1 = self.conv_1x1(x)
        x2 = self.conv_3x3_1(x)
        x3 = self.conv_3x3_2(x)
        x4 = self.conv_3x3_3(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x