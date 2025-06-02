import os
import sys
sys.path.append(os.getcwd())

import torch.nn as nn
from src.network.modules import CBAM

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ConvBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
    def forward(self, x):
        return self.conv_block(x)

class EncodeELA(nn.Module):
    def __init__(self):
        super(EncodeELA, self).__init__()
        self.conv1 = ConvBlock(1,64)
        self.cbam1 = CBAM(64)
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv2 = ConvBlock(64,128)
        self.cbam2 = CBAM(128)
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv3 = ConvBlock(128,256)
        self.cbam3 = CBAM(256)
        self.pool3 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv4 = ConvBlock(256,512)
        self.pool4 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.cbam4 = CBAM(512)
        self.conv5 = ConvBlock(512,512)
        self.pool5 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.cbam5 = CBAM(512)
        

    def forward(self, x):
        conv1 = self.conv1(x)
        cbam1 = self.cbam1(conv1)
        pool1 = self.pool1(cbam1) 
        #print(pool1.shape)
        conv2 = self.conv2(pool1)
        cbam2 = self.cbam2(conv2)
        pool2 = self.pool2(cbam2)
        #print(pool2.shape)
        conv3 = self.conv3(pool2)
        cbam3 = self.cbam3(conv3)
        pool3 = self.pool3(cbam3)
        #print(pool3.shape)
        conv4 = self.conv4(pool3)
        cbam4 = self.cbam4(conv4)
        pool4 = self.pool4(cbam4)
        #print(pool4.shape)
        conv5 = self.conv5(pool4)
        cbam5 = self.cbam5(conv5)
        pool5 = self.pool5(cbam5)
        #print(pool5.shape)
        
        return pool1, pool2, pool3, pool4, pool5