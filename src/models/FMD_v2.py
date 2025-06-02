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
    def __init__(self, device):
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
        self.criterion_cls = nn.CrossEntropyLoss()
        # self.criterion_rec = VGG19PerceptureLoss()
        
    def forward(self):
        # get output of u2net-gan
        self.d0, self.d1, self.d2, self.d3, self.d4, self.d5, self.pred_label = self.u2net_gan_v2(self.inputs, self.ela)

        return self.d0, self.d1, self.d2, self.d3, self.d4, self.d5, self.pred_label
        
    def set_input(self, inputs, segment_labels, ela=None, cls_labels=None, real_images=None):
        self.inputs = inputs.to(self.device)
        self.segment_labels = segment_labels.to(self.device)
        self.ela = ela.to(self.device)
        self.cls_labels = cls_labels.to(self.device)
        self.real_images = real_images.to(self.device) if real_images is not None else real_images
        
    def set_requires_grad(self, nets, requires_grad=True):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
            
    
    def backward_u2net_gan_v2(self, training=True, lambda_seg=20, lambda_cls=1):
        """ calculate loss for u2net-gan
        """
        
        # segmentation loss
        loss_seg_d0 = self.criterion_segment(self.d0, self.segment_labels)
        loss_seg_d1 = self.criterion_segment(self.d1, self.segment_labels)
        loss_seg_d2 = self.criterion_segment(self.d2, self.segment_labels)
        loss_seg_d3 = self.criterion_segment(self.d3, self.segment_labels)
        loss_seg_d4 = self.criterion_segment(self.d4, self.segment_labels)
        loss_seg_d5 = self.criterion_segment(self.d5, self.segment_labels)

        self.loss_seg = loss_seg_d0 +  loss_seg_d1 + 0.4 * (loss_seg_d2 + loss_seg_d3 + loss_seg_d4 + loss_seg_d5)
        
        # classification loss
        self.loss_cls = self.criterion_cls(self.pred_label, self.cls_labels)

        # combine loss
        total_loss = self.loss_seg * lambda_seg + lambda_cls * self.loss_cls
        if training:
            total_loss.backward()
        
        return self.loss_seg * lambda_seg, lambda_cls * self.loss_cls
    
    
    def optimize_parameters(self, freeze_cls_branch=False): 
        """ optimizer for u2net-gan
        """
        # freeze cls branch
        if freeze_cls_branch:
            for name, param in self.u2net_gan_v2.named_parameters():
                if 'classifier' in name or 'encode_ela' in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        else:
            for name, param in self.u2net_gan_v2.named_parameters():
                param.requires_grad = True
    
        # pass data into network
        self.forward()
        # train u2net-gan
        self.optimizer_u2net_gan_v2.zero_grad()
        self.loss_seg, self.loss_cls = self.backward_u2net_gan_v2()
        self.optimizer_u2net_gan_v2.step()
        
        return self.loss_seg, self.loss_cls
    
    def get_loss(self):
        return self.loss_seg, self.loss_cls
    
    def get_num_true_pred_images(self):
        predicted_classes = torch.argmax(self.pred_label, dim=1)  # shape: [N]

        true_classes = self.cls_labels.view(-1)
    
        correct = (predicted_classes == true_classes).sum().item()
        total = true_classes.size(0)
        return correct, total