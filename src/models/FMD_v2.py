import os
import sys
import cv2
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
from torch import optim

from tqdm import tqdm
from src.network.backbone_gan import *
from src.network.u2net_gan_v2 import init_u2net_gan_v2

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
        self.optimizer_u2net_gan_v2 = optim.Adam(self.u2net_gan_v2.parameters(), lr=0.001, betas=(0.9, 0.999))
        
        # scheduler
        
        # loss function
        self.criterion_segment = nn.L1Loss()
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
        
        loss_seg = 2 * loss_d0 + loss_d1 + 0.4 * (loss_d2 + loss_d3 + loss_d4 + loss_d5 + loss_d6)

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