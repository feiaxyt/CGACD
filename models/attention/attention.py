import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config.config import cfg

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, z_f, x_f):
        raise NotImplementedError

class PixelAttention(Attention):
    def __init__(self, feat_in=256):
        super(PixelAttention, self).__init__()
        self.feat_in = feat_in
        
        self.spatial_pool_agl = nn.Sequential(
            nn.Conv2d(25, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 3),
            nn.Sigmoid(),
        )

        self.spatial_pool_agr = nn.Sequential(
            nn.Conv2d(25, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 3),
            nn.Sigmoid(),
        )

        self.channel_pool_ag = nn.Sequential(
            nn.Linear(feat_in, feat_in//4),
            nn.ReLU(inplace=True),
            nn.Linear(feat_in//4, feat_in),
        )
        
        self.channel_maxpool = nn.MaxPool2d(cfg.train.search_pool_size - cfg.train.template_pool_size + 1)
        self.channel_avgpool = nn.AvgPool2d(cfg.train.search_pool_size - cfg.train.template_pool_size + 1)
        self.channel_activation = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    m.weight.data, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

    def forward(self, z_f, x_f):
        b, c, h, w = z_f.shape
        kernel = z_f.reshape(b,c,h*w).permute(0,2,1).reshape(-1, c, 1, 1)
        b, c, h, w = x_f.shape
        xf_reshape = x_f.reshape(1, -1, h, w)
        pixel_corr = F.conv2d(xf_reshape, kernel, groups=b).reshape(b, -1, h, w)# / c
        b, c, h, w = pixel_corr.shape
        spatial_att_l = self.spatial_pool_agl(pixel_corr)
        spatial_att_r = self.spatial_pool_agr(pixel_corr)
        b, c, h, w = z_f.shape
        kernel = z_f.reshape(b*c, 1, h, w)
        b, c, h, w = x_f.shape
        xf_reshape = x_f.reshape(1, b*c, h, w)
        depth_corr = F.conv2d(xf_reshape, kernel, groups=b*c)
        depth_corr = depth_corr.reshape(b, c, depth_corr.shape[-2], depth_corr.shape[-1])
        channel_max_pool = self.channel_maxpool(depth_corr).squeeze()
        channel_avg_pool = self.channel_avgpool(depth_corr).squeeze()
        channel_att = self.channel_activation(self.channel_pool_ag(channel_max_pool) + self.channel_pool_ag(channel_avg_pool)).unsqueeze(-1).unsqueeze(-1)
        
        x_f = x_f * channel_att
        x_f_l = x_f * spatial_att_l
        x_f_r = x_f * spatial_att_r
        return x_f_l, x_f_r