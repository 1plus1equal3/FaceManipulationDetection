import torch
import torch.nn as nn

def init_discriminator():
    return Discriminator(ConvDownBlock, in_channels=3)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, down_sample=True):
        """ Convolutional Block of GAN Encoder

        Args:
            in_channels (_type_): in_channels of block
            out_channels (_type_): out_channels of block
        """
        super(ConvBlock, self).__init__()
        self.down_sample = down_sample
        if self.down_sample:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        else:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
            
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
    
class Discriminator(nn.Module):
    def __init__(self, ConvDownBlock, in_channels=3):
        """ discriminator class for CycleGAN using PatchGAN

        Args:
            ConvDownBlock: Convolution downsampling block
            in_channels (int, optional): Defaults to 3.
            num_conv_blocks (int, optional): Number of conv down blocks. Defaults to 8.
        """
        
        super(Discriminator, self).__init__()
        
        conv_downs = []
        self.middle_channels = 64
        for i in range(5):
            conv_downs.append(ConvDownBlock(
                in_channels = in_channels,
                out_channels = self.middle_channels,
                kernel_size = 4,
                stride = 2 if i < 3 else 1,
                padding = 0,
                act='leaky_relu',
                norm='instance' if i != 4 else 'none'
            ))
            in_channels = self.middle_channels
            self.middle_channels = self.middle_channels * 2
            
        # Final layer is 1 channel
        conv_downs.append(ConvDownBlock(in_channels, 1, kernel_size=4, stride=1, padding=1, act='sigmoid', norm='none'))
        self.model = nn.Sequential(*conv_downs)
        
    def forward(self, x):
        x = self.model(x)
        return x
    
class ConvDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, act='leaky_relu', norm='instance'):
        super(ConvDownBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        
        # Normalization layer
        if norm == 'instance':
            self.norm = nn.InstanceNorm2d(out_channels, affine=False, track_running_stats=False)
        elif norm == 'batch':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'none':
            self.norm = nn.Identity()
        else:
            raise ValueError("Unsupported normalization type: {}".format(norm))

        # Activation function
        if act == 'leaky_relu':
            self.act = nn.LeakyReLU(0.2, inplace=True)
        elif act == 'tanh':
            self.act = nn.Tanh()
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            raise ValueError("Unsupported activation function: {}".format(act))
    
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))