import os
import sys
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
from torchsummary import summary
from src.network.backbone_u2_net import *
from src.network.backbone_gan import *

def _upsample_like(src,tar):

    src = F.interpolate(src,size=tar.shape[2:],mode='bilinear')

    return src

def init_u2net_gan():
    return U2Net_GAN(in_ch=3, out_ch=1)

class U2Net_GAN(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(U2Net_GAN, self).__init__()
        
        # encoder
        self.stage1 = RSU7(in_ch,32,64)
        self.encoder_1 = ConvBlock(3, 64, kernel_size=7, stride=1, padding=3, down_sample=True)
        self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage2 = RSU6(64,32,128)
        self.res_from_encoder_2 = ConvBlock(64, 128, kernel_size=3, stride=1, padding=1, down_sample=True)
        self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage3 = RSU5(128,64,256)
        self.res_from_encoder_3 = ConvBlock(128, 256, kernel_size=3, stride=1, padding=1, down_sample=True)
        self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage4 = RSU4(256,128,512)
        self.res_from_encoder_4 = ConvBlock(256, 512, kernel_size=3, stride=1, padding=1, down_sample=True)
        self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage5 = RSU4F(512,256,512)
        self.res_from_encoder_5 = ConvBlock(512, 512, kernel_size=3, stride=1, padding=1, down_sample=True)
        self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage6 = RSU4F(512,256,512)
        # gan bottle neck
        self.bottle_neck = ConvBlock(512, 512, kernel_size=3, stride=1, padding=1)

        # decoder
        # self.stage5d = RSU4F(1024,256,512)
        self.stage4d = RSU4(1024,128,256)
        self.res_from_decoder_4 = ConvBlock(1024, 256, kernel_size=4, stride=2, padding=1, down_sample=False)
        self.stage3d = RSU5(512,64,128)
        self.res_from_decoder_3 = ConvBlock(512, 128, kernel_size=4, stride=2, padding=1, down_sample=False)
        self.stage2d = RSU6(256,32,64)
        self.res_from_decoder_2 = ConvBlock(256, 64, kernel_size=4, stride=2, padding=1, down_sample=False)
        self.stage1d = RSU7(128,16,64)
        self.res_from_decoder_1 = ConvBlock(128, 3, kernel_size=4, stride=2, padding=1, down_sample=False, act='tanh')

        self.side1 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side2 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side3 = nn.Conv2d(128,out_ch,3,padding=1)
        self.side4 = nn.Conv2d(256,out_ch,3,padding=1)
        self.side5 = nn.Conv2d(512,out_ch,3,padding=1)
        self.side6 = nn.Conv2d(512,out_ch,3,padding=1)

        self.outconv = nn.Conv2d(5*out_ch,out_ch,1)

    def forward(self,x):

        hx = x
        encode = x

        #stage 1
        hx1 = self.stage1(hx)   # u2net: (3->64)
        encoder_1 = self.encoder_1(encode)  # encoder: (3->64)
        hx = self.pool12(hx1)
        encode = self.pool12(encoder_1)

        #stage 2
        hx2 = self.stage2(hx + encode)   # u2net: (64->128) and combine information from encoder to u2net
        encoder_2 = self.res_from_encoder_2(hx + encode)    # gan: combine inf from u2net and prev encoder
        hx = self.pool23(hx2)            # pooling
        encode = self.pool23(encoder_2)
        
        #stage 3
        hx3 = self.stage3(hx + encode)
        encoder_3 = self.res_from_encoder_3(hx + encode)
        hx = self.pool34(hx3)
        encode = self.pool34(encoder_3)

        #stage 4
        hx4 = self.stage4(hx + encode)
        encoder_4 = self.res_from_encoder_4(hx + encode)
        hx = self.pool45(hx4)
        encode = self.pool45(encoder_4)

        # #stage 5
        # hx5 = self.stage5(hx + encode)
        # encoder_5 = self.res_from_encoder_5(hx + encode)
        # hx = self.pool56(hx5)
        # encode = self.pool56(encoder_5)

        #stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6,hx4)
        
        # GAN bottle next
        bottle_neck = self.bottle_neck(encode)

        #-------------------- decoder --------------------
        # hx5d = self.stage5d(torch.cat((hx6up,hx5),1))
        # hx5dup = _upsample_like(hx5d,hx4)

        decoder_4 = self.res_from_decoder_4(torch.cat((hx6, bottle_neck), 1))
        hx4d = self.stage4d(torch.cat((hx6up, hx4), 1))
        hx4dup = _upsample_like(hx4d,hx3)

        decoder_3 = self.res_from_decoder_3(torch.cat((hx4d, decoder_4), 1))
        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d,hx2)

        decoder_2 = self.res_from_decoder_2(torch.cat((hx3d, decoder_3), 1))
        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d,hx1)

        decoder_1 = self.res_from_decoder_1(torch.cat((hx2d, decoder_2), 1))
        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))
        # reconstruction image
        
        
        # side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2,d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3,d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4,d1)

        # d5 = self.side5(hx5d)
        # d5 = _upsample_like(d5,d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6,d1)

        d0 = self.outconv(torch.cat((d1,d2,d3,d4,d6),1))

        return decoder_1, F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d6)
    
u2net_gan = U2Net_GAN()
summary(u2net_gan, (3, 256, 256))