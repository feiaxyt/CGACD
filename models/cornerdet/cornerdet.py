import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CornerDet(nn.Module):
    def __init__(self):
        super(CornerDet, self).__init__()

    def forward(self, x_f):
        raise NotImplementedError


class SepCornerDet(CornerDet):
    def __init__(self, feat_in=256):
        super(SepCornerDet, self).__init__()

        self.up0_l = nn.Sequential(
            nn.Conv2d(feat_in, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.up1_l = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.up2_l = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )

        self.up0_r = nn.Sequential(
            nn.Conv2d(feat_in, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.up1_r = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.up2_r = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    m.weight.data, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

    def forward(self, x_f):
        x_f_l, x_f_r = x_f
        x_f_l = self.up0_l(x_f_l)
        resolution = x_f_l.shape[-1]
        x_f_l = self.up1_l(F.interpolate(
            x_f_l, size=(resolution*2+1, resolution*2+1)))
        resolution = x_f_l.shape[-1]
        heat_map_l = self.up2_l(F.interpolate(
            x_f_l, size=(resolution*2+1, resolution*2+1)))

        batch_sz = x_f_l.shape[0]
        left_top_map = F.softmax(heat_map_l.squeeze().reshape(batch_sz, -1), 1).reshape(
            batch_sz, heat_map_l.shape[-2], heat_map_l.shape[-1])

        x_f_r = self.up0_r(x_f_r)
        resolution = x_f_r.shape[-1]
        x_f_r = self.up1_r(F.interpolate(
            x_f_r, size=(resolution*2+1, resolution*2+1)))
        resolution = x_f_r.shape[-1]
        heat_map_r = self.up2_r(F.interpolate(
            x_f_r, size=(resolution*2+1, resolution*2+1)))
        batch_sz = x_f_r.shape[0]
        right_bottom_map = F.softmax(heat_map_r.squeeze().reshape(batch_sz, -1), 1).reshape(
            batch_sz, heat_map_r.shape[-2], heat_map_r.shape[-1])

        heatmap_size = left_top_map.shape[-1]
        xx, yy = np.meshgrid([dx for dx in range(int(heatmap_size))],
                             [dy for dy in range(int(heatmap_size))])
        heatmap_xx = torch.from_numpy(xx).float().cuda()
        heatmap_yy = torch.from_numpy(yy).float().cuda()

        x1 = ((left_top_map * heatmap_xx).sum(-1).sum(-1) /
              heatmap_xx.shape[-1]).reshape(-1, 1)
        y1 = ((left_top_map * heatmap_yy).sum(-1).sum(-1) /
              heatmap_xx.shape[-2]).reshape(-1, 1)
        x2 = ((right_bottom_map * heatmap_xx).sum(-1).sum(-1) /
              heatmap_xx.shape[-1]).reshape(-1, 1)
        y2 = ((right_bottom_map * heatmap_yy).sum(-1).sum(-1) /
              heatmap_xx.shape[-2]).reshape(-1, 1)

        result_target = torch.cat((x1, y1, x2, y2), 1)

        return result_target, left_top_map.shape[-1]