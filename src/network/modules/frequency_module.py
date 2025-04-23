import torch
import torch.nn as nn
import torch.nn.functional as F

class FrequencyModule(nn.Module):
    """ calculate and filter high-pass frequency to create an attention map

    Args:
        x (batch_size, channels, height, width): input
    Returns:
        high frequency domain in range [0, 1]
    """
    def __init__(self):
        super(FrequencyModule, self).__init__()
        self.conv_freq_1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.conv_freq_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        self.conv_freq_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        self.sigmoid = nn.Sigmoid()

    def high_pass_filter(self, x, size):
        y, x = torch.meshgrid(torch.linspace(-size//2, size//2, size), torch.linspace(-size//2, size//2, size))
        x = x.to(x.device)
        y = y.to(x.device)
        dist = torch.sqrt(x**2 + y**2)
        filter = 1 - torch.exp(-(dist**2) / (2 * 5.0**2))  # High-pass filter
        return filter.unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        fft = torch.fft.fft2(x, dim=(-2, -1))           # (B, 3, H, W)
        fft_real = fft.real
        fft_imag = fft.imag
        fft_input = torch.cat([fft_real, fft_imag], dim=1)      # (B, 6, H, W)
        high_filter = self.high_pass_filter(x, height).to(x.device)
        fft_high = fft_input * high_filter              # apply high pass filter
        
        # apply conv for frequency module
        freq_1 = self.conv_freq_1(fft_high)
        freq_2 = self.conv_freq_2(freq_1)
        freq_3 = self.conv_freq_3(freq_2)
        
        # frequency1 = self.conv_freq(fft_high)           # (B, 1, 256, 256)
        # frequency2 = F.interpolate(frequency1, size=(frequency1.size(2) // 2, frequency1.size(3) // 2), mode='bilinear')    # (B, 1, 128, 128)
        # frequency3 = F.interpolate(frequency1, size=(frequency1.size(2) // 4, frequency1.size(3) // 4), mode='bilinear')    # (B, 1, 64, 64)
        
        return freq_1, freq_2, freq_3                # (B, 1, H, W)