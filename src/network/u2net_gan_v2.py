import os
import sys
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from src.network.modules import CBAM, FrequencyModule, TextureModule, \
                            AttentionGate, AdaptiveFusion, TripletAttention, PyramidPoolingModule
from src.network.backbone_u2_net import *

## upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src,tar):

    src = F.interpolate(src,size=tar.shape[2:],mode='bilinear')

    return src

def init_u2net_gan_v2():
    return U2NetGanV2()


##### U^2-Net ####
class U2NetGanV2(nn.Module):
    def __init__(self,in_ch=3,out_ch=1):
        super(U2NetGanV2,self).__init__()
        
        # Frequency Module
        self.frequency = FrequencyModule()
        
        # U2 Net
        self.stage1 = RSU7(in_ch,32,64)
        # CBAM
        self.cbam1 = CBAM(64)
        # Triplet attention
        self.triplet1 = TripletAttention(no_spatial=False)
        # Texture Module
        self.text1 = TextureModule(64)
        # Adaptive Fusion
        self.adapt_fusion1 = AdaptiveFusion(64, 64)
        # Attention Gate
        self.attn_gate_1 = AttentionGate(64)
        self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        
        
        self.stage2 = RSU6(64,32,128)
        # CBAM
        self.cbam2 = CBAM(128)
        # Triplet attention
        self.triplet2 = TripletAttention(no_spatial=False)
        # Texture Module
        self.text2 = TextureModule(128)
        # Adaptive Fusion
        self.adapt_fusion2 = AdaptiveFusion(128, 128)
        # Attention Gate
        self.attn_gate_2 = AttentionGate(128)
        self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)


        self.stage3 = RSU5(128,64,256)
        # CBAM
        self.cbam3 = CBAM(256)
        # Triplet Attention
        self.triplet3 = TripletAttention(no_spatial=False)
        # Texture Module
        self.text3 = TextureModule(256)
        # Adaptive Fusion
        self.adapt_fusion3 = AdaptiveFusion(256, 256)
        # Attention Gate
        self.attn_gate_3 = AttentionGate(256)
        self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage4 = RSU4(256,128,512)
        # CBAM
        self.cbam4 = CBAM(512)
        # Triplet Attention
        self.triplet4 = TripletAttention(no_spatial=False)
        # # Texture Module
        # self.text4 = TextureModule(512)
        # # Adaptive Fusion
        # self.adapt_fusion1 = AdaptiveFusion(512, 512)
        self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage5 = RSU4F(512,256,512)
        # Triplet Attention
        self.triplet5 = TripletAttention(no_spatial=False)
        
        self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage6 = RSU4F(512,256,512)
        # Pyramid Pooling
        self.pyramid = PyramidPoolingModule(512, 512)

        # decoder
        self.stage5d = RSU4F(1024,256,512)
        self.stage4d = RSU4(1024,128,256)
        self.stage3d = RSU5(512,64,128)
        self.stage2d = RSU6(256,32,64)
        self.stage1d = RSU7(128,16,64)
        
        self.side1 = nn.Conv2d(64,out_ch,3,padding=1)      # concat with frequency module
        self.side2 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side3 = nn.Conv2d(128,out_ch,3,padding=1)
        self.side4 = nn.Conv2d(256,out_ch,3,padding=1)
        self.side5 = nn.Conv2d(512,out_ch,3,padding=1)
        self.side6 = nn.Conv2d(512,out_ch,3,padding=1)

        self.outconv = nn.Conv2d(6*out_ch,out_ch,1)

    def forward(self,x):
        
        # frequency module
        frequency1, frequency2, frequency3 = self.frequency(x)

        hx = x

        #stage 1
        hx1 = self.stage1(hx)
        hx1 = self.triplet1(hx1)        # triplet attention
        hx = self.pool12(hx1)

        #stage 2
        hx2 = self.stage2(hx)
        hx2 = self.triplet2(hx2)        # triplet attention
        hx = self.pool23(hx2)

        #stage 3
        hx3 = self.stage3(hx)
        hx3 = self.triplet3(hx3)        # triplet attention
        hx = self.pool34(hx3)

        #stage 4
        hx4 = self.stage4(hx)
        hx4 = self.triplet4(hx4)        # triplet attention
        hx = self.pool45(hx4)

        #stage 5
        hx5 = self.stage5(hx)
        hx5 = self.triplet5(hx5)        # triplet attention
        hx = self.pool56(hx5)

        #stage 6
        hx6 = self.stage6(hx)
        hx6 = self.pyramid(hx6)         # pyramid pooling
        hx6up = _upsample_like(hx6,hx5)

        #-------------------- decoder --------------------
        hx5d = self.stage5d(torch.cat((hx6up,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.stage4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.stage3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.stage2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.stage1d(torch.cat((hx2dup,hx1),1))


        # side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2,d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3,d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4,d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5,d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6,d1)

        d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6),1))

        return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)