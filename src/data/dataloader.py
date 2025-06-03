import json
import os
import sys

import cv2
import random
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.getcwd())

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import transforms
from src.utils.util import high_pass_filter_image

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
        
        # fft high pass filter
        ela = high_pass_filter_image(image_path)
        
        # self.label_paths is not None => fake_image
        if self.label_paths:
            # fake
            true_cls_label = 1
            # read mask image
            label_path = self.label_paths[idx]
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            
            # for kaggle path
            # parent_dir = os.path.dirname(image_path)
            # real_image_path = (parent_dir + '/' + os.path.basename(image_path).split('_')[0] + '_0.' + image_path.split('.')[1]).replace('fake_attrGAN/fake_attrGAN', 'real-20250326T031740Z-001/real')
            
            # for local path
            real_image_path = (image_path.split('_')[0] + '_0.' + image_path.split('.')[1]).replace('fakes', 'reals')
            real_image = cv2.imread(real_image_path)
            real_image = cv2.cvtColor(real_image, cv2.COLOR_BGR2RGB)

            # true_cls_label = torch.tensor([true_cls_label], dtype=torch.float32)
            
            if self.trans_input:
                image = self.trans_input(image)
                ela = self.trans_input(ela)
                # real_image = self.trans_input(real_image)
                
            if self.trans_label:
                label = self.trans_label(label)
        
            return image, label, ela, true_cls_label
        
        # real_image
        else:
            # real
            true_cls_label = 0
            # segment label
            label =  np.zeros(image.shape[:2], dtype=np.float32) # (H, W)
            label = np.expand_dims(label, axis=2)         # (H, W, 1)

            # true_cls_label = torch.tensor([true_cls_label], dtype=torch.float32)
            
            if self.trans_input:
                image = self.trans_input(image)
                ela = self.trans_input(ela)
                
            if self.trans_label:
                label = self.trans_label(label)
            
            return image, label, ela, true_cls_label
    

def get_data_paths(dataset_path):
    # real + fake + mask image paths
    # for local path
    real_image_folder_path = os.path.join(dataset_path, 'reals')
    fake_image_folder_path = os.path.join(dataset_path, 'fakes')
    mask_image_folder_path = os.path.join(dataset_path, 'masks')

    # for kaggle path
    # real_image_folder_path = os.path.join(dataset_path, 'dataset-attrgan/real-20250326T031740Z-001/real')
    # fake_image_folder_path = os.path.join(dataset_path, 'dataset-attrgan/fake_attrGAN/fake_attrGAN')
    # mask_image_folder_path = os.path.join(dataset_path, 'masked-dataset-newversion/mask')

    # get real image paths
    real_image_paths = sorted([os.path.join(real_image_folder_path, real_image_path) for real_image_path in os.listdir(real_image_folder_path)])
    # get fake image paths
    fake_image_paths = sorted([os.path.join(fake_image_folder_path, fake_image_path) for fake_image_path in os.listdir(fake_image_folder_path)])
    # get mask image paths
    mask_image_paths = sorted([os.path.join(mask_image_folder_path, mask_image_path) for mask_image_path in os.listdir(mask_image_folder_path)])

    # return path of real, fake and mask image paths
    return real_image_paths, fake_image_paths, mask_image_paths
    

def get_dataloader(dataset_path, mode='train', type='fake', attribute=None):
    random.seed(42)  # for reproducibility
    real_image_paths, fake_image_paths, mask_image_paths = get_data_paths(dataset_path=dataset_path)
    
    # train model with 30000 fake images and 3000 real images
    len_dataset = len(real_image_paths)
    random_fake_paths = random.sample(fake_image_paths, k=len_dataset)
    random_mask_paths = [mask_image_paths[fake_image_paths.index(path)] for path in random_fake_paths]
    fake_image_paths = random_fake_paths
    mask_image_paths = random_mask_paths

    # train test split
    train_real_image_paths, val_real_image_paths = train_test_split(real_image_paths, test_size=0.3, random_state=42)
    
    train_fake_image_paths, val_fake_image_paths, train_mask_images_paths, \
        val_mask_image_paths = train_test_split(fake_image_paths, mask_image_paths, test_size=0.3, random_state=42)

    # get attribute data
    if attribute == 'bald':
        bald_loader = get_bald_data(val_fake_image_paths, val_mask_image_paths)
        return bald_loader
    elif attribute == 'eyeglass':
        eyeglass_loader = get_eyeglass_data(val_fake_image_paths, val_mask_image_paths)
        return eyeglass_loader
    elif attribute == 'smile':
        smile_loader = get_smile_data(val_fake_image_paths, val_mask_image_paths)
        return smile_loader
    else:
        print(f"Training with no attribute")

    if mode == 'train':
        # real dataset
        train_real_dataset = GANDataset_V2(
                # train with small part of data for testing
                image_paths=train_real_image_paths,
                trans_input=trans_input,
                trans_label=trans_label
            )

        train_real_loader = DataLoader(train_real_dataset, batch_size=config['datasets']['train'], shuffle=True)
        
        # fake dataset
        train_fake_dataset = GANDataset_V2(
                image_paths=train_fake_image_paths,
                label_paths=train_mask_images_paths,
                trans_input=trans_input,
                trans_label=trans_label
            )

        train_fake_loader = DataLoader(train_fake_dataset, batch_size=config['datasets']['train'], shuffle=True)
        
        if type == 'real':
            return train_real_loader
        elif type == 'fake':
            return train_fake_loader
        elif type == 'combined':
            combined_dataset = ConcatDataset([train_real_dataset, train_fake_dataset])
            combined_loader = DataLoader(combined_dataset, batch_size=config['datasets']['train'], shuffle=True)
            return combined_loader
        else:
            raise ValueError(f"Invalid dataset type: {type}. Must be 'real', 'fake', or 'combined'")
            
    elif mode == 'test':
        # real dataset
        test_real_dataset = GANDataset_V2(
                image_paths=val_real_image_paths,
                trans_input=trans_input,
                trans_label=trans_label
            )
            
        test_real_loader = DataLoader(test_real_dataset, batch_size=config['datasets']['val'], shuffle=True)
        
        # fake dataset
        test_fake_dataset = GANDataset_V2(
                image_paths=val_fake_image_paths,
                label_paths=val_mask_image_paths,
                trans_input=trans_input,
                trans_label=trans_label
            )
            
        test_fake_loader = DataLoader(test_fake_dataset, batch_size=config['datasets']['val'], shuffle=True)
            
        if type == 'real':
            return test_real_loader
        elif type == 'fake':
            return test_fake_loader
        elif type == 'combined':
            combined_dataset = ConcatDataset([test_real_dataset, test_fake_dataset])
            combined_loader = DataLoader(combined_dataset, batch_size=config['datasets']['val'], shuffle=True)
            return combined_loader
        else:
            raise ValueError(f"Invalid dataset type: {type}. Must be 'real', 'fake', or 'combined'")

def get_bald_data(dataset, label):
    bald_dataset_path = []
    bald_label_path = []
    for i, path in enumerate(dataset):
        if os.path.basename(path).split('_')[1] == '1.jpg':
            bald_dataset_path.append(path)
            bald_label_path.append(label[i])

    bald_dataset = GANDataset_V2(bald_dataset_path, bald_label_path, trans_input=trans_input, trans_label=trans_label)
    bald_loader = DataLoader(bald_dataset, batch_size=config['datasets']['val'], shuffle=True)
    return bald_loader

def get_eyeglass_data(dataset, label):
    eyeglass_dataset_path = []
    eyeglass_label_path = []
    for i, path in enumerate(dataset):
        if os.path.basename(path).split('_')[1] == '7.jpg':
            eyeglass_dataset_path.append(path)
            eyeglass_label_path.append(label[i])

    eyeglass_dataset = GANDataset_V2(eyeglass_dataset_path, eyeglass_label_path, trans_input=trans_input, trans_label=trans_label)
    eyeglass_loader = DataLoader(eyeglass_dataset, batch_size=config['datasets']['val'], shuffle=True)
    return eyeglass_loader

def get_smile_data(dataset, label):
    smile_dataset_path = []
    smile_label_path = []
    for i, path in enumerate(dataset):
        if os.path.basename(path).split('_')[1] == '9.jpg':
            smile_dataset_path.append(path)
            smile_label_path.append(label[i])
            
    smile_dataset = GANDataset_V2(smile_dataset_path, smile_label_path, trans_input=trans_input, trans_label=trans_label)
    smile_loader = DataLoader(smile_dataset, batch_size=config['datasets']['val'], shuffle=True)
    return smile_loader