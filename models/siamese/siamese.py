# Copyright (c) SenseTime. All Rights Reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F


def xcorr_depthwise(x, kernel):
    """depthwise cross correlation
    """
    batch = kernel.size(0)
    channel = kernel.size(1)
    x = x.view(1, batch*channel, x.size(2), x.size(3))
    kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))
    out = F.conv2d(x, kernel, groups=batch*channel)
    out = out.view(batch, channel, out.size(2), out.size(3))
    return out


def xcorr_up(x, kernel):
    batch_sz = kernel.shape[0]
    kernel = kernel.reshape(-1, x.shape[1],
                            kernel.shape[2], kernel.shape[3])
    out = F.conv2d(
        x.reshape(1, -1, x.shape[2], x.shape[3]), kernel, groups=batch_sz)
    out = out.reshape(batch_sz, -1, out.shape[2], out.shape[3])
    return out


class UPXCorr(nn.Module):
    def __init__(self, out_channels, adjust, feat_in=256, feat_out=256):
        super(UPXCorr, self).__init__()
        self.conv_kernel = nn.Conv2d(feat_in, feat_out * out_channels, 3)
        self.conv_search = nn.Conv2d(feat_in, feat_out, 3)
        if adjust:
            self.adjust = nn.Conv2d(out_channels, out_channels, 1)
        else:
            self.adjust = lambda x: x

    def forward(self, z_f, x_f):
        kernel = self.conv_kernel(z_f)
        search = self.conv_search(x_f)
        out = xcorr_up(search, kernel)
        return self.adjust(out)


class DepthwiseXCorr(nn.Module):
    def __init__(self, feat_in=256, feat_out=256, out_channels=1, kernel_size=3):
        super(DepthwiseXCorr, self).__init__()
        self.conv_kernel = nn.Sequential(
            nn.Conv2d(feat_in, feat_out, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(feat_out),
            nn.ReLU(inplace=True),
        )
        self.conv_search = nn.Sequential(
            nn.Conv2d(feat_in, feat_out, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(feat_out),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(feat_out, feat_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(feat_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_out, out_channels, kernel_size=1)
        )
        self.kernel = None

    def forward(self, kernel, search):
        kernel = self.conv_kernel(kernel)
        search = self.conv_search(search)
        feature = xcorr_depthwise(search, kernel)
        out = self.head(feature)
        return out


class Siamese(nn.Module):
    def __init__(self):
        super(Siamese, self).__init__()

    def forward(self, z_f, x_f):
        raise NotImplementedError


def normal_init(m, mean, stddev):
    m.weight.data.normal_(mean, stddev)
    m.bias.data.zero_()


class UPChannelSiamese(Siamese):
    def __init__(self, feat_in=256, feature_out=256):
        super(UPChannelSiamese, self).__init__()
        self.cls = UPXCorr(1, False, feat_in, feature_out)
        self.loc = UPXCorr(4, True, feat_in, feature_out)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, 0, 0.001)

    def forward(self, z_f, x_f):
        loc = self.loc(z_f[:,:,4:-4,4:-4], x_f)
        cls = self.cls(z_f[:,:,4:-4,4:-4], x_f)
        return loc, cls


class DepthwiseSiamese(Siamese):
    def __init__(self, feat_in=256, feature_out=256):
        super(DepthwiseSiamese, self).__init__()
        self.cls = DepthwiseXCorr(feat_in, feature_out, 1)
        self.loc = DepthwiseXCorr(feat_in, feature_out, 4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight.data, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

    def forward(self, z_f, x_f):
        cls = self.cls(z_f, x_f)
        loc = self.loc(z_f, x_f)
        return loc, cls