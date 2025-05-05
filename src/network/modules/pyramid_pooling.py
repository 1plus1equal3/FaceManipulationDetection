import torch
import torch.nn as nn
import torch.nn.functional as F

# Pyramid Pooling Module (PPM)
class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, out_channels, pool_sizes=(1, 2, 3, 6)):
        super(PyramidPoolingModule, self).__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=size),
                nn.Conv2d(in_channels, out_channels // 4, 1, bias=False),
                nn.BatchNorm2d(out_channels // 4),
                nn.ReLU(inplace=True)
            ) for size in pool_sizes
        ])
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + (out_channels // 4) * len(pool_sizes), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.dropout = nn.Dropout2d(0.3)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        feats = [x]
        for stage in self.stages:
            y = stage(x)
            y = F.interpolate(y, size=(h, w), mode='bilinear', align_corners=True)
            feats.append(y)
        x = torch.cat(feats, dim=1)
        x = self.conv(x)
        x = self.dropout(x)
        return x