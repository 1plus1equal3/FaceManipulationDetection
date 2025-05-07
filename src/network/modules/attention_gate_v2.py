import torch.nn as nn

class AttentionGateV2(nn.Module):
    def __init__(self,channel = 3):
        super(AttentionGateV2, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, kernel_size = 1, stride = 1)
        self.conv2 = nn.ConvTranspose2d(channel, channel, kernel_size = 2, stride = 2)
        self.conv_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size = 3, stride = 1, padding = 1),
            nn.Sigmoid()
        )
    def forward(self, x, ela):
        x1 = self.conv1(x)
        #print(x1.shape)
        ela1 = self.conv2(ela)
        #print(ela1.shape)
        attn_gate = self.conv_block(x1 + ela1)
        return x * attn_gate