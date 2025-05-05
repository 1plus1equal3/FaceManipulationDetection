import json
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.getcwd())

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# transform for input image
trans_input = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# transform for gt mask
trans_label = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256))
])

# load config file
with open('src/config.json', 'r') as f:
    config = json.load(f)

class GanDataset(Dataset):
    def __init__(self, real_image_paths, fake_image_paths, masked_fake_image_paths, trans_input=None, trans_label=None):
        self.real_image_paths = real_image_paths
        self.fake_image_paths = fake_image_paths
        self.masked_fake_image_paths = masked_fake_image_paths
        self.trans_input = trans_input
        self.trans_label = trans_label

    def __len__(self):
        # combine real and fake for training
        return len(self.fake_image_paths) + len(self.real_image_paths)

    def __getitem__(self, idx):
        if idx < len(self.real_image_paths):
            image = cv2.imread(self.real_image_paths[idx])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            masked_image = np.zeros(image.shape[:2], dtype=np.uint8) # (H, W)
            masked_image = np.expand_dims(masked_image, axis=2)  # (H, W, 1)
            input_is_real = True
        else:
            image = cv2.imread(self.fake_image_paths[idx - len(self.real_image_paths)])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            masked_image = cv2.imread(self.masked_fake_image_paths[idx - len(self.real_image_paths)], cv2.IMREAD_GRAYSCALE)
            input_is_real = False

        # trans input
        if self.trans_input:
            image = self.trans_input(image)
            
        # trans label    
        if self.trans_label:
            masked_image = self.trans_label(masked_image)

        return image, masked_image, input_is_real
    
    
class GANDataset_V2(Dataset):
    def __init__(self, image_paths, label_paths=None, trans_input=None, trans_label=None):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.trans_input = trans_input
        self.trans_label = trans_label
            
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # self.label_paths is not None => fake_image
        if self.label_paths:
            # read mask image
            label_path = self.label_paths[idx]
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            
            # for kaggle path
            parent_dir = os.path.dirname(image_path)
            real_image_path = (parent_dir + '/' + os.path.basename(image_path).split('_')[0] + '_0.' + image_path.split('.')[1]).replace('fake_attrGAN/fake_attrGAN', 'real-20250326T031740Z-001/real')
            
            # for local path
            # real_image_path = (image_path.split('_')[0] + '_0.' + image_path.split('.')[1]).replace('fake_attrGAN/fake_attrGAN', 'real-20250326T031740Z-001/real')
            real_image = cv2.imread(real_image_path)
            real_image = cv2.cvtColor(real_image, cv2.COLOR_BGR2RGB)
            
            if self.trans_input:
                image = self.trans_input(image)
                real_image = self.trans_input(real_image)
                
            if self.trans_label:
                label = self.trans_label(label)
        
            return image, label, real_image
        
        # real_image
        else:
            label =  np.zeros(image.shape[:2], dtype=np.uint8) # (H, W)
            label = np.expand_dims(label, axis=2)         # (H, W, 1)
        
            if self.trans_input:
                image = self.trans_input(image)
                
            if self.trans_label:    
                label = self.trans_label(label)
            
            return image, label
    

def get_data_paths(dataset_path):
    # real + fake + mask image paths
    real_image_folder_path = os.path.join(dataset_path, 'dataset-attrgan/real-20250326T031740Z-001/real')
    fake_image_folder_path = os.path.join(dataset_path, 'dataset-attrgan/fake_attrGAN/fake_attrGAN')
    mask_image_folder_path = os.path.join(dataset_path, 'masked-dataset-newversion/mask')

    # get real image paths
    real_image_paths = sorted([os.path.join(real_image_folder_path, real_image_path) for real_image_path in os.listdir(real_image_folder_path)])
    # get fake image paths
    fake_image_paths = sorted([os.path.join(fake_image_folder_path, fake_image_path) for fake_image_path in os.listdir(fake_image_folder_path)])
    # get mask image paths
    mask_image_paths = sorted([os.path.join(mask_image_folder_path, mask_image_path) for mask_image_path in os.listdir(mask_image_folder_path)])

    # return path of real, fake and mask image paths
    return real_image_paths, fake_image_paths, mask_image_paths
    

def get_dataloader(dataset_path, mode='train', is_real=False):
    real_image_paths, fake_image_paths, mask_image_paths = get_data_paths(dataset_path=dataset_path)
    
    # train model with 30000 fake images and 3000 real images
    real_image_paths = real_image_paths[:3000]
    mask_image_paths = mask_image_paths[:30000]
    fake_image_paths = fake_image_paths[:30000]

    # train test split
    train_real_image_paths, val_real_image_paths = train_test_split(real_image_paths, test_size=0.3, random_state=42)
    
    train_fake_image_paths, val_fake_image_paths, train_mask_images_paths, \
        val_mask_image_paths = train_test_split(fake_image_paths, mask_image_paths, test_size=0.3, random_state=42)

    if mode == 'train':
        if is_real:
            train_real_dataset = GANDataset_V2(
                # train with small part of data for testing
                image_paths=train_real_image_paths,
                trans_input=trans_input,
                trans_label=trans_label
            )
        
            # sampler = DistributedSampler(train_real_dataset)
            train_real_loader = DataLoader(train_real_dataset, batch_size=config['datasets']['is_real']['train'], shuffle=True)
            return train_real_loader
        else:
            train_fake_dataset = GANDataset_V2(
                image_paths=train_fake_image_paths,
                label_paths=train_mask_images_paths,
                trans_input=trans_input,
                trans_label=trans_label
            )
            
            # sampler = DistributedSampler(train_fake_dataset)
            train_fake_loader = DataLoader(train_fake_dataset, batch_size=config['datasets']['not_is_real']['train'], shuffle=True)
            return train_fake_loader
    elif mode == 'test':
        if is_real:
            test_real_dataset = GANDataset_V2(
                image_paths=val_real_image_paths,
                trans_input=trans_input,
                trans_label=trans_label
            )
            
            # sampler = DistributedSampler(test_real_dataset)
            test_real_loader = DataLoader(test_real_dataset, batch_size=config['datasets']['is_real']['val'], shuffle=True)
            return test_real_loader
        else:
            test_fake_dataset = GANDataset_V2(
                image_paths=val_fake_image_paths,
                label_paths=val_mask_image_paths,
                trans_input=trans_input,
                trans_label=trans_label
            )
            
            # sampler = DistributedSampler(test_fake_dataset)
            test_fake_loader = DataLoader(test_fake_dataset, batch_size=config['datasets']['not_is_real']['val'], shuffle=True)
            return test_fake_loader