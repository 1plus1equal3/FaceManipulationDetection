import torch
import torch.nn as nn

class TextureModule(nn.Module):
    def __init__(self, in_channels):
        super(TextureModule, self).__init__()
        
        # conv module
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels)
        )
        
    def forward(self, x):
        texture_block = x
        texture_block = self.conv(texture_block)
        
        return texture_block