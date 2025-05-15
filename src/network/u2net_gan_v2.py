import os
import sys
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from src.network.modules import AttentionGateV2, EncodeELA, Classifier
from src.network.backbone_u2_net import *
from src.utils import init_weights

## upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src,tar):

    src = F.interpolate(src,size=tar.shape[2:],mode='bilinear')

    return src

def init_u2net_gan_v2():
    model = U2NetGanV2()
    init_weights(model)

    return model


##### U^2-Net ####
class U2NetGanV2(nn.Module):
    def __init__(self,in_ch=3,out_ch=1):
        super(U2NetGanV2,self).__init__()
        
        # Encode ELA
        self.encode_ela = EncodeELA()
        self.gate1 = AttentionGateV2(512)
        self.gate2 = AttentionGateV2(512)
        self.gate3 = AttentionGateV2(256)
        self.gate4 = AttentionGateV2(128)
        self.gate5 = AttentionGateV2(64)
        
        # U2 Net
        self.stage1 = RSU7(in_ch,32,64)
        self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        
        
        self.stage2 = RSU6(64,32,128)
        self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)


        self.stage3 = RSU5(128,64,256)
        self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage4 = RSU4(256,128,512)
        self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage5 = RSU4F(512,256,512)
        
        self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)

        self.stage6 = RSU4F(512,256,512)

        # decoder
        self.stage5d = RSU4F(1024,256,512)
        self.stage4d = RSU4(1024,128,256)
        self.stage3d = RSU5(512,64,128)
        self.stage2d = RSU6(256,32,64)
        self.stage1d = RSU7(128,16,64)
        
        self.side1 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side2 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side3 = nn.Conv2d(128,out_ch,3,padding=1)
        self.side4 = nn.Conv2d(256,out_ch,3,padding=1)
        self.side5 = nn.Conv2d(512,out_ch,3,padding=1)
        self.side6 = nn.Conv2d(512,out_ch,3,padding=1)

        self.outconv = nn.Conv2d(6*out_ch,out_ch,1)
        self.classifier = Classifier(num_classes=2)

    def forward(self,x,ela):
        #--------------------encode------------------------
        hx = x

        #stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        #stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        #stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        #stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        #stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        #stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6,hx5)
        
        #-----------------------ela-----------------------
        ela1, ela2, ela3, ela4, ela5 = self.encode_ela(ela)
        
        #--------------------classifier-------------------
        pred = self.classifier(ela5)
        
        #------------------attention gate-----------------
        attn5 = self.gate1(hx5, ela5)
        attn4 = self.gate2(hx4, ela4)
        attn3 = self.gate3(hx3, ela3)
        attn2 = self.gate4(hx2, ela2)
        attn1 = self.gate5(hx1, ela1)

        #-------------------- decoder --------------------
        hx5d = self.stage5d(torch.cat((hx6up,attn5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.stage4d(torch.cat((hx5dup,attn4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.stage3d(torch.cat((hx4dup,attn3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.stage2d(torch.cat((hx3dup,attn2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.stage1d(torch.cat((hx2dup,attn1),1))


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

        d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6), 1))

        return torch.sigmoid(d0), torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(d4), torch.sigmoid(d5), torch.sigmoid(d6), pred
    
# x = torch.randn(1, 3, 256, 256)
# y = torch.randn(1, 3, 256, 256)
# model = U2NetGanV2()
# result = model(x, y)
# print(result[0].shape)  # Output shape of the first output (d0)

# num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"Number of trainable parameters: {num_params}") 

# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name, param.data.shape)