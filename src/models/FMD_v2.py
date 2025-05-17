import os
import sys
import json
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
from torch import optim

from tqdm import tqdm
from src.network.backbone_gan import *
from src.network.u2net_gan_v2 import init_u2net_gan_v2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

config = load_config('src/config.json')

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1.0):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        
        pos_weight = torch.tensor([5.0]).to(device)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.smooth = smooth
 
    def forward(self, logits, targets):
        # BCE Loss
        bce = self.bce(logits, targets)

        # Dice Loss
        probs = torch.sigmoid(logits)
        targets = targets.type_as(probs)

        intersection = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))

        dice = 1 - (2. * intersection + self.smooth) / (union + self.smooth)
        dice = dice.mean()

        return self.bce_weight * bce + self.dice_weight * dice

class FMD_v2(nn.Module):
    def __init__(self, device, in_channels=3):
        super(FMD_v2, self).__init__()
        self.device = device
        
        # init model
        self.u2net_gan_v2 = init_u2net_gan_v2()
        
        # use_gpu
        self.u2net_gan_v2.to(self.device)
        
        # define for training
        # optim
        self.optimizer_u2net_gan_v2 = optim.Adam(self.u2net_gan_v2.parameters(), lr=1e-4, betas=(0.9, 0.999))
        
        # scheduler
        
        # loss function
        self.criterion_segment = BCEDiceLoss()
        # self.criterion_rec = VGG19PerceptureLoss()
        
    def forward(self):
        # get output of u2net-gan
        self.d0, self.d1, self.d2, self.d3, self.d4, self.d5, self.d6 = self.u2net_gan_v2(self.inputs)

        return self.d0, self.d1, self.d2, self.d3, self.d4, self.d5, self.d6
        
    def set_input(self, inputs, labels, real_images=None, input_is_real=None):
        self.inputs = inputs.to(self.device)
        self.labels = labels.to(self.device)
        self.real_images = real_images.to(self.device) if real_images is not None else real_images
        self.input_is_real = input_is_real
        
    def set_requires_grad(self, nets, requires_grad=True):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
            
    
    def backward_u2net_gan_v2(self, training=True):
        """ calculate loss for u2net-gan
        """
        
        # segmentation loss
        loss_d0 = self.criterion_segment(self.d0, self.labels)
        loss_d1 = self.criterion_segment(self.d1, self.labels)
        loss_d2 = self.criterion_segment(self.d2, self.labels)
        loss_d3 = self.criterion_segment(self.d3, self.labels)
        loss_d4 = self.criterion_segment(self.d4, self.labels)
        loss_d5 = self.criterion_segment(self.d5, self.labels)
        loss_d6 = self.criterion_segment(self.d6, self.labels)
        
        loss_seg = loss_d0 + loss_d1 + 0.4 * (loss_d2 + loss_d3 + loss_d4 + loss_d5 + loss_d6)

        if training:
            loss_seg.backward()
        
        return loss_seg
    
    
    def optimize_parameters(self):
        # pass data into network
        self.forward()
        # train u2net-gan
        self.optimizer_u2net_gan_v2.zero_grad()
        loss_seg = self.backward_u2net_gan_v2()
        self.optimizer_u2net_gan_v2.step()
        
        return loss_seg
    
    def calculate_loss(self, pred, gt):
        pred = pred.to(self.device)
        gt = gt.to(self.device)
        
        loss = self.criterion_segment(pred, gt)
        return loss