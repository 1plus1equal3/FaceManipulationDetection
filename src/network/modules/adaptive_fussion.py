import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveFusion(nn.Module):
    def __init__(self, in_channels_texture, in_channels_rgb):
        super(AdaptiveFusion, self).__init__()
        self.weight_layer = nn.Conv2d(in_channels_rgb * 2, 2, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_channels_rgb)
    
    def forward(self, texture_features, rgb_features):
        combined = torch.cat([texture_features, rgb_features], dim=1)  # [B, in_channels_rgb * 2, H, W]
        weights = torch.softmax(self.weight_layer(combined), dim=1)  # [B, 2, H, W]
        w_texture, w_rgb = weights[:, 0:1, :, :], weights[:, 1:2, :, :]
        spatial_features = w_texture * texture_features + w_rgb * rgb_features  # [B, in_channels_rgb, H, W]
        spatial_features = self.bn(spatial_features)
        return spatial_features