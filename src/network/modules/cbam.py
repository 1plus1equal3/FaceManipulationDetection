import torch
import torch.nn as nn
import torch.nn.functional as F

# CBAM
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.channel_sigmoid = nn.Sigmoid()
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.spatial_sigmoid = nn.Sigmoid()

    def forward(self, x, freq_attn=None):
        avg_pool = self.channel_avg_pool(x)
        max_pool = self.channel_max_pool(x)
        avg_out = self.channel_mlp(avg_pool)
        max_out = self.channel_mlp(max_pool)
        channel_attn = self.channel_sigmoid(avg_out + max_out)
        x = x * channel_attn
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        spatial_attn = self.spatial_sigmoid(self.spatial_conv(spatial_input))
        if freq_attn is not None:
            freq_attn = F.interpolate(freq_attn, size=x.shape[2:], mode='bilinear')
            spatial_attn = spatial_attn * freq_attn
        x = x * spatial_attn
        return x