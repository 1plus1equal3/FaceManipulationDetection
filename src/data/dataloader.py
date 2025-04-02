import os
import sys
import cv2
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

dataset_path = '/kaggle/input/dataset-attrgan'
masked_path = '/kaggle/input/masked-dataset-newversion'

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
        return len(self.real_image_paths)

    def __getitem__(self, idx):
        real_image = cv2.imread(self.real_image_paths[idx])
        real_image = cv2.cvtColor(real_image, cv2.COLOR_BGR2RGB)

        fake_image = cv2.imread(self.fake_image_paths[idx])
        fake_image = cv2.cvtColor(fake_image, cv2.COLOR_BGR2RGB)

        masked_fake_image = cv2.imread(self.masked_fake_image_paths[idx], cv2.IMREAD_GRAYSCALE)

        if self.trans:
            real_image = self.trans(real_image)
            fake_image = self.trans(fake_image)
            masked_fake_image = self.trans(masked_fake_image)
            masked_real_image = torch.zeros_like(masked_fake_image, dtype=real_image.dtype)

        return real_image, masked_real_image, fake_image, masked_fake_image

def get_data_paths(dataset_path, masked_path):
    # real + fake + mask image paths
    real_image_folder_path = os.path.join(dataset_path, 'real-20250326T031740Z-001/real')
    fake_image_folder_path = os.path.join(dataset_path, 'fake_attrGAN/fake_attrGAN')
    mask_image_folder_path = os.path.join(masked_path, 'mask')

    # get real image paths
    real_image_paths = sorted([os.path.join(real_image_folder_path, real_image_path) for real_image_path in os.listdir(real_image_folder_path)])
    # get fake image paths
    fake_image_paths = sorted([os.path.join(fake_image_folder_path, fake_image_path) for fake_image_path in os.listdir(fake_image_folder_path)])
    # get mask image paths
    mask_image_paths = sorted([os.path.join(mask_image_folder_path, mask_image_path) for mask_image_path in os.listdir(mask_image_folder_path)])

    # return path of real, fake and mask image paths
    return real_image_paths, fake_image_paths, mask_image_paths
    

def get_dataloader(mode='train'):
    real_image_paths, fake_image_paths, mask_image_paths = get_data_paths(
        dataset_path=dataset_path,
        masked_path=masked_path
    )
    
    if mode == 'train':
        train_dataset = GanDataset(
            real_image_paths=real_image_paths,
            fake_image_paths=fake_image_paths,
            masked_fake_image_paths=mask_image_paths,
            trans=trans
        )
        
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=False)
    
    return train_dataloader