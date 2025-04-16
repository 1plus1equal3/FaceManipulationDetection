import os
import sys
import torch
import torch.nn as nn
import torchvision

from torchvision import models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Using percepture loss from vgg19
class VGG19PerceptureLoss(nn.Module):
    """ calculate perceptural loss using vgg19 model

    Args:
        image1(torch.Tensor): image
        image2(torch.Tensor): image
    """
    def __init__(self, device=device):
        super(VGG19PerceptureLoss, self).__init__()
        self.device = device
        
        # define model VGG19
        VGG19_pretrained = models.vgg19(weights='IMAGENET1K_V1').features
        # define the layer in vgg19 (relu_1, relu_2, relu_3, relu_4)
        self.layer1 = nn.Sequential(*VGG19_pretrained[:4]).to(self.device)
        self.layer2 = nn.Sequential(*VGG19_pretrained[4:9]).to(self.device)
        self.layer3 = nn.Sequential(*VGG19_pretrained[9:16]).to(self.device)
        self.layer4 = nn.Sequential(*VGG19_pretrained[16:23]).to(self.device)
        # Freeze all the vgg19
        for param in self.parameters():
            param.requires_grad = False

        # loss
        self.loss = nn.L1Loss()

    def forward(self, image1, image2):
        image1 = image1.to(self.device)
        image2 = image2.to(self.device)
        
        # norm input follow vgg19
        image1, image2 = self.norm_follow_vgg19(image1, image2)

        # extracted image1
        hx = image1
        image1_h1 = self.layer1(hx)
        hx = image1_h1
        image1_h2 = self.layer2(hx)
        hx = image1_h2
        image1_h3 = self.layer3(hx)
        hx = image1_h3
        image1_h4 = self.layer4(hx)

        # extracted image2
        hx = image2
        image2_h1 = self.layer1(hx)
        hx = image2_h1
        image2_h2 = self.layer2(hx)
        hx = image2_h2
        image2_h3 = self.layer3(hx)
        hx = image2_h3
        image2_h4 = self.layer4(hx)

        # # calculate style loss
        # style_loss = 0.0
        # image1_layers = [image1_h1, image1_h2, image1_h3, image1_h4]
        # image2_layers = [image2_h1, image2_h2, image2_h3, image2_h4]
        # for img1, img2 in zip(image1_layers, image2_layers):
        #     style_loss += self.compute_style_loss(img1, img2)

        # return style_loss

        # calculate perceptural loss
        perceptural_loss = 0.0
        image1_layers = [image1_h1, image1_h2, image1_h3, image1_h4]
        image2_layers = [image2_h1, image2_h2, image2_h3, image2_h4]
        for img1, img2 in zip(image1_layers, image2_layers):
            perceptural_loss += self.loss(img1, img2)

        return perceptural_loss

    def norm_follow_vgg19(self, image1, image2):
        # rescale [-1, 1] -> [0, 1]
        image1 = ((image1 + 1) / 2)
        image2 = ((image2 + 1) / 2)

        # normalize follow mean and std of image net
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(self.device)
        imagenet_std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(self.device)

        image1 = (image1 - imagenet_mean) / imagenet_std
        image2 = (image2 - imagenet_mean) / imagenet_std

        return image1, image2

    def gram_matrix(self, features):
        b, c, h, w = features.size()
        features = features.view(b*c, -1)
        features = features
        return torch.mm(features, features.t()) / (b * c * h * w)

    def compute_style_loss(self, matrix1, matrix2):
        gram_matrix_1 = self.gram_matrix(matrix1)
        gram_matrix_2 = self.gram_matrix(matrix2)
    
        # calculate loss
        style_loss = self.loss(gram_matrix_1, gram_matrix_2)
        return style_loss 
    