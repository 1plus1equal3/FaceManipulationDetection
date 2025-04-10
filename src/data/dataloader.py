import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

dataset_path = '/datasets'

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

def get_data_paths(dataset_path):
    # real + fake + mask image paths
    real_image_folder_path = os.path.join(dataset_path, 'real-20250326T031740Z-001/real')
    fake_image_folder_path = os.path.join(dataset_path, 'fake_attrGAN/fake_attrGAN')
    mask_image_folder_path = os.path.join(dataset_path, 'mask')

    # get real image paths
    real_image_paths = sorted([os.path.join(real_image_folder_path, real_image_path) for real_image_path in os.listdir(real_image_folder_path)])
    # get fake image paths
    fake_image_paths = sorted([os.path.join(fake_image_folder_path, fake_image_path) for fake_image_path in os.listdir(fake_image_folder_path)])
    # get mask image paths
    mask_image_paths = sorted([os.path.join(mask_image_folder_path, mask_image_path) for mask_image_path in os.listdir(mask_image_folder_path)])

    # return path of real, fake and mask image paths
    return real_image_paths, fake_image_paths, mask_image_paths
    

def get_dataloader(mode='train'):
    real_image_paths, fake_image_paths, mask_image_paths = get_data_paths(dataset_path=dataset_path)
    
    if mode == 'train':
        train_dataset = GanDataset(
            # train with small part of data for testing
            real_image_paths=real_image_paths[:2000],
            fake_image_paths=fake_image_paths[:20000],
            masked_fake_image_paths=mask_image_paths,
            trans=trans
        )
        
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
        return train_dataloader
    elif mode == 'test':
        test_dataset = GanDataset(
            real_image_paths=real_image_paths[3000:3100],
            fake_image_paths=fake_image_paths[30000:31000],
            masked_fake_image_paths=mask_image_paths,
            trans=trans
        )
        
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
        return test_loader