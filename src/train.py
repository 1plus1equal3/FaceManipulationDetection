import os
import sys
import json

sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.optim as optim

from src.data.dataloader import get_dataloader
from src.utils.util import *
from src.models.FMD_v2 import *
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

config = load_config('src/config.json')

def main():
    train_real_loader = get_dataloader(mode='train', is_real=True)
    train_fake_loader = get_dataloader(mode='train', is_real=False)
    test_real_loader = get_dataloader(mode='test', is_real=True)
    test_fake_loader = get_dataloader(mode='test', is_real=False)
    
    # define model
    fmd_v2 = FMD_v2(device=device)
    
    # phase 1: train with real image
    epochs = config['model']['epoch']
    for epoch in tqdm(range(30)):
        total_loss_seg = 0.0
        
        for (input, true_mask) in (train_real_loader):
            fmd_v2.set_input(input, true_mask)
            loss_seg = fmd_v2.optimize_parameters()
            
            # total loss
            total_loss_seg += loss_seg
            
        if (epoch + 1) % 10 == 0 or epoch == 0:
            
            # save checkpoint
            if not os.path.exists('checkpoints'):
                os.makedirs('checkpoints')
                
            torch.save(fmd_v2.state_dict(), 'checkpoints/weights_v2.pt')
            
            # visualize some sample in test loader
            total_loss_seg_val = 0.0
            with torch.no_grad():
                for i, (input, true_mask) in enumerate(test_real_loader):
                    fmd_v2.set_input(input, true_mask)
                    pred_mask, _, _, _, _, _, _ = fmd_v2()
                    
                    if i == 20:
                        visualize_results(input, input, true_mask, pred_mask, epoch+1, text='phase_1')
                    
                    loss = fmd_v2.calculate_loss(pred_mask, true_mask)
                    total_loss_seg_val += loss.items()
            
            print(f"loss_seg_train: {total_loss_seg/len(train_real_loader):.4f}\t loss_seg_val: {total_loss_seg_val/len(test_real_loader)}")
                
                
    # phase 2: train with fake image
    epochs = config['model']['epoch']
    for epoch in tqdm(range(100)):
        total_loss_seg = 0.0

        for (input, true_mask, real_image) in train_fake_loader:
            fmd_v2.set_input(input, true_mask, real_image)
            loss_seg = fmd_v2.optimize_parameters()
            
            # total loss
            total_loss_seg += loss_seg
            
        if (epoch + 1) % 10 == 0 or epoch == 0:
            
            # save checkpoint
            if not os.path.exists('checkpoints'):
                os.makedirs('checkpoints')
                
            torch.save(fmd_v2.state_dict(), 'checkpoints/weights_attn_gate.pt')
            
            # visualize some sample in test loader
            total_loss_seg_val = 0.0
            with torch.no_grad():
                for input, true_mask, real_image in test_fake_loader:
                    fmd_v2.set_input(input, true_mask, real_image)
                    pred_mask, _, _, _, _, _, _ = fmd_v2()
                    
                    if i == 20:
                        visualize_results(input, real_image, true_mask, pred_mask, epoch+1, text='phase_2')
                    
                    loss = fmd_v2.calculate_loss(pred_mask, true_mask)
                    total_loss_seg_val += loss.items()
            
            print(f"loss_seg_train: {total_loss_seg/len(train_real_loader):.4f}\t loss_seg_val: {total_loss_seg_val/len(test_real_loader)}")
                

if __name__ == '__main__':
    main()