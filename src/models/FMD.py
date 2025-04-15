import os
import sys
import cv2
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
from torch import optim
import torchvision

from tqdm import tqdm
from src.network.backbone_gan import *
from src.network.u2net_gan import *
from src.utils.perceptural_loss import *

class FMD(nn.Module):
    def __init__(self, device, in_channels=3):
        super(FMD, self).__init__()
        self.device = device
        
        # init model
        self.u2net_gan = init_u2net_gan()
        self.disc = init_discriminator()
        
        # set training mode
        self.u2net_gan.train()
        self.disc.train()
        
        # use_gpu
        self.u2net_gan.to(self.device)
        self.disc.to(self.device)
        
        # define for training
        # optim
        self.optimizer_u2net_gan = optim.Adam(self.u2net_gan.parameters(), lr=0.001, betas=(0.5, 0.999))
        self.optimizer_disc = optim.Adam(self.disc.parameters(), lr=0.001, betas=(0.5, 0.999))
        
        # scheduler
        
        # loss function
        self.criterion_segment = nn.BCELoss(size_average=True)
        self.criterion_disc = nn.MSELoss()
        self.criterion_gan = VGG19PerceptureLoss()
        self.criterion_rec = nn.L1Loss()
        
    def forward(self):
        # get output of u2net-gan
        self.rec_img, self.d0, self.d1, self.d2, self.d3, self.d4, self.d6 = self.u2net_gan(self.inputs)

        return self.rec_img, self.d0, self.d1, self.d2, self.d3, self.d4, self.d6
        
    def set_input(self, inputs, labels, real_images=None, input_is_real=None):
        self.inputs = inputs.to(self.device)
        self.labels = labels.to(self.device)
        self.real_images = real_images
        self.input_is_real = input_is_real.to(self.device)
    
    def get_label_from_input(self, input, input_is_real):
        if input_is_real:
            return torch.zeros_like(input)
        else:
            return torch.ones_like(input)
        
    def set_requires_grad(self, nets, requires_grad=True):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
            
    
    def backward_u2net_gan(self):
        """ calculate loss for u2net-gan
        """
        
        # segmentation loss
        loss_d0 = self.criterion_segment(self.d0, self.labels)
        loss_d1 = self.criterion_segment(self.d1, self.labels)
        loss_d2 = self.criterion_segment(self.d2, self.labels)
        loss_d3 = self.criterion_segment(self.d3, self.labels)
        loss_d4 = self.criterion_segment(self.d4, self.labels)
        loss_d6 = self.criterion_segment(self.d6, self.labels)
        
        loss_seg = loss_d0 + loss_d1 + loss_d2 + loss_d3 + loss_d4 + loss_d6
        
        # gan loss (D(G(x)) -> 1)
        loss_gen = self.criterion_gan(self.disc(self.rec_img), torch.ones_like(self.disc(self.rec_img)))
        
        # reconstruct loss
        if self.real_images is not None:
            loss_rec = self.criterion_rec(self.rec_img, self.real_images)
        else:
            loss_rec = 0
        
        # conbine loss u2net-gan
        loss_u2net_gan =loss_seg + loss_gen + loss_rec
        loss_u2net_gan.backward()
        
        return loss_seg, loss_gen, loss_rec
    
    def backward_disc(self):
        """ calculate loss for discriminator
        """

        # dis criminator loss
        pred_rec = self.disc(self.rec_img.detach())      # pass rec into discriminator
        
        # rec_label = self.get_label_from_input(pred_rec, self.input_is_real)
        # input_label = self.get_label_from_input(self.inputs, True)
        
        loss_dis = (self.criterion_disc(pred_rec, torch.ones_like(pred_rec)) + self.criterion_disc(self.inputs, torch.zeros_like(self.inputs))) * 0.5
        
        loss_dis.backward()
        return self.criterion_disc(pred_rec, torch.ones_like(pred_rec)), self.criterion_disc(self.inputs, torch.zeros_like(self.inputs))
    
    def optimize_parameters(self):
        # pass data into network
        self.forward()
        # train u2net-gan
        self.optimizer_u2net_gan.zero_grad()
        self.set_requires_grad([self.disc], False)      # freeze disc when training u2net-gan
        loss_seg, loss_gen, loss_rec = self.backward_u2net_gan()
        self.optimizer_u2net_gan.step()
        
        # train disc
        self.optimizer_disc.zero_grad()
        self.set_requires_grad([self.disc], True)
        loss_D_fake, loss_D_real = self.backward_disc()
        self.optimizer_disc.step()
        
        return loss_seg, loss_gen, loss_rec, loss_D_fake, loss_D_real
    
    def optimize_segmentation(self):
        # pass data into network
        self.forward()
        # train u2net
        self.optimizer_u2net_gan.zero_grad()
        self.set_requires_grad([self.disc], False)
        loss_seg, _, _ = self.backward_u2net_gan()
        self.optimizer_u2net_gan.step()
        
        return loss_seg