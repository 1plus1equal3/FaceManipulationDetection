import os
import sys
import cv2
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# Get image masks
def get_images_masked(mask_folder_path):
    image_masked_list = []

    # read mask folder and get the masked images
    mask_folder_paths = sorted([entry.name for entry in os.scandir(mask_folder_path) if entry.is_file()])
    for mask_filename in tqdm(mask_folder_paths):
        if mask_filename.split('_')[0] not in image_masked_list:
            image_masked_list.append(mask_filename.split('_')[0])
        else:
            continue

    return sorted(image_masked_list)


# Function to generate mask from a pair or real and fake image
def soft_ratio(x, p=2):
    max_val = np.max(x)
    result = (x ** p) / (x ** p + (max_val - x) ** p)
    return result


def exponential(x, p=2):
    return np.exp(x)


def get_mask(real_image, fake_image, apply_math='soft_ratio', threshold=0.1):
    # perform mask
    real_image_fl = np.array(real_image, dtype=np.float32)
    fake_image_fl = np.array(fake_image, dtype=np.float32)

    mask = np.abs(real_image_fl - fake_image_fl)
    mask = np.max(mask, axis=2)

    if apply_math == 'soft_ratio':
        mask = soft_ratio(mask)
        mask = np.where(mask < threshold, 0, mask)
    elif apply_math == 'exponential':
        mask = exponential(mask)
        mask = np.where(mask < threshold, 0, mask)

    return mask

def save_image(mask, save_mask_folder, fake_image_filename):
    mask = np.round(mask * 255.0).astype(np.uint8)
    cv2.imwrite(os.path.join(save_mask_folder, fake_image_filename), mask)

def gen_mask(image_masked_list, fake_folder_path, real_folder_path, save_mask_folder):
    need_check = True
    need_read_real_image = True
    
    if not os.path.exists(save_mask_folder):
        os.makedirs(save_mask_folder)

    # fake_folder_paths = sorted(glob.glob(fake_folder_path + '/*'))
    fake_folder_paths = sorted([entry.name for entry in os.scandir(fake_folder_path) if entry.is_file()])
    for fake_image_filename in tqdm(fake_folder_paths):

        if need_check:
            if fake_image_filename.split('_')[0] in image_masked_list:
                continue
            else:
                need_check = False

        fake_image_name = fake_image_filename.split('_')[0]
        fake_image_attribute = fake_image_filename.split('_')[1]

        if need_read_real_image:
            real_image = cv2.imread(os.path.join(real_folder_path, fake_image_name + '_0.jpg'))
            need_read_real_image = False

        # start of a fake image with different attribute
        if fake_image_attribute == '1.jpg':
            real_image = cv2.imread(os.path.join(real_folder_path, fake_image_name + '_0.jpg'))
            fake_image = cv2.imread(os.path.join(fake_folder_path, fake_image_filename))
            mask = get_mask(real_image, fake_image, apply_math='soft_ratio')
            need_read_real_image = True

            save_image(mask, save_mask_folder, fake_image_filename)
        else:
            fake_image = cv2.imread(os.path.join(fake_folder_path, fake_image_filename))
            mask = get_mask(real_image, fake_image, apply_math='soft_ratio')

            save_image(mask, save_mask_folder, fake_image_filename)
            

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--real_path', type=str, default='', help='Path to your real image folder')
    parser.add_argument('--fake_path', type=str, default='', help='Path to your fake image folder')
    parser.add_argument('--save_path', type=str, default='', help='Path to your save image folder')
    
    args = parser.parse_args()
    
    gen_mask(
        image_masked_list=get_images_masked(args.save_path),
        fake_folder_path=args.fake_path,
        real_folder_path=args.real_path,
        save_mask_folder=args.save_path
    )
    
if __name__ == '__main__':
    main()