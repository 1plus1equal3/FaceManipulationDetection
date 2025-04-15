import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

dataset_path = 'datasets'

# transform
trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
])

class GanDataset(Dataset):
    def __init__(self, real_image_paths, fake_image_paths, masked_fake_image_paths, trans=None):
        self.real_image_paths = real_image_paths
        self.fake_image_paths = fake_image_paths
        self.masked_fake_image_paths = masked_fake_image_paths
        self.trans = trans

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

        if self.trans:
            image = self.trans(image)
            masked_image = self.trans(masked_image)

        return image, masked_image, input_is_real
    
    
class GANDataset_V2(Dataset):
    def __init__(self, image_paths, label_paths=None, trans=None):
        self.image_paths = image_paths
        self.trans = trans
        self.label_paths = label_paths
    
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
            label = cv2.imread(label_path)
            
            real_image_path = (image_path.split('_')[0] + '_0.' + image_path.split('.')[1]).replace('fake', 'real')
            real_image = cv2.imread(real_image_path)
            
            if self.trans:
                image = trans(image)
                label = trans(label)
                real_image = trans(real_image)
        
            return image, label, real_image
        # real_image
        else:
            label =  np.zeros(image.shape[:2], dtype=np.uint8) # (H, W)
            label = np.expand_dims(label, axis=2)         # (H, W, 1)
        
            if self.trans:
                image = trans(image)
                label = trans(label)
            
            return image, label
    

def get_data_paths(dataset_path):
    # real + fake + mask image paths
    real_image_folder_path = os.path.join(dataset_path, 'reals')
    fake_image_folder_path = os.path.join(dataset_path, 'fakes')
    mask_image_folder_path = os.path.join(dataset_path, 'masks')

    # get real image paths
    real_image_paths = sorted([os.path.join(real_image_folder_path, real_image_path) for real_image_path in os.listdir(real_image_folder_path)])
    # get fake image paths
    fake_image_paths = sorted([os.path.join(fake_image_folder_path, fake_image_path) for fake_image_path in os.listdir(fake_image_folder_path)])
    # get mask image paths
    mask_image_paths = sorted([os.path.join(mask_image_folder_path, mask_image_path) for mask_image_path in os.listdir(mask_image_folder_path)])

    # return path of real, fake and mask image paths
    return real_image_paths, fake_image_paths, mask_image_paths
    

def get_dataloader(mode='train', is_real=False):
    real_image_paths, fake_image_paths, mask_image_paths = get_data_paths(dataset_path=dataset_path)
    
    if mode == 'train':
        if is_real:
            train_real_dataset = GANDataset_V2(
                # train with small part of data for testing
                image_paths=real_image_paths[:2000],
                trans=trans
            )
        
            train_real_loader = DataLoader(train_real_dataset, batch_size=32, shuffle=True)
            return train_real_loader
        else:
            train_fake_dataset = GANDataset_V2(
                image_paths=fake_image_paths[:20000],
                label_paths=mask_image_paths[:20000],
                trans=trans
            )
            
            train_fake_loader = DataLoader(train_fake_dataset, batch_size=32,shuffle=True)
            return train_fake_loader
    elif mode == 'test':
        if is_real:
            test_real_dataset = GANDataset_V2(
                real_image_paths=real_image_paths[3000:3100],
                trans=trans
            )
            
            test_real_loader = DataLoader(test_real_dataset, batch_size=16, shuffle=True)
            return test_real_loader
        else:
            test_fake_dataset = GANDataset_V2(
                image_paths=fake_image_paths[30000:31000],
                label_paths=mask_image_paths[30000:31000],
                trans=trans
            )
            
            test_fake_loader = DataLoader(test_fake_dataset, batch_size=32, shuffle=False)
            return test_fake_loader