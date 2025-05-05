import json
import os
import sys
import argparse

sys.path.append(os.getcwd())

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from src.data.dataloader import get_dataloader
from src.models.FMD_v2 import *
from src.utils.util import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

config = load_config('src/config.json')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='', help='path to your dataset')
    parser.add_argument('--save_results', type=str, default='', help='path to save results')
    
    args = parser.parse_args()
    
    # define model
    fmd_v2 = FMD_v2(device=device)
    
    # dataloader
    train_real_loader = get_dataloader(args.dataset_path, mode='train', is_real=True)
    train_fake_loader = get_dataloader(args.dataset_path, mode='train', is_real=False)
    test_real_loader = get_dataloader(args.dataset_path, mode='test', is_real=True)
    test_fake_loader= get_dataloader(args.dataset_path, mode='test', is_real=False)
    
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    
    # phase 1: train with real image
    # epochs = config['model']['epoch']
    # for epoch in tqdm(range(10)):
    #     total_loss_seg = 0.0
        
    #     for (input, true_mask) in (train_real_loader):
    #         fmd_v2.set_input(input, true_mask)
    #         loss_seg = fmd_v2.optimize_parameters()
            
    #         # total loss
    #         total_loss_seg += loss_seg
            
    #     if (epoch + 1) % 10 == 0 or epoch == 0:
            
    #         # save checkpoint
    #         if not os.path.exists('checkpoints'):
    #             os.makedirs('checkpoints')
                
    #         if torch.distributed.get_rank() == 0:
    #             # Save model's weight
    #             checkpoint_path = f'checkpoints/weights_attn_gate.pt'
    #             torch.save(fmd_v2.state_dict(), checkpoint_path)
            
    #         # visualize some sample in test loader
    #         print(len(test_real_loader))
    #         total_loss_seg_val = 0.0
    #         with torch.no_grad():
    #             for i, (input, true_mask) in enumerate(test_real_loader):
    #                 fmd_v2.set_input(input, true_mask)
    #                 pred_mask, _, _, _, _, _, _ = fmd_v2()
                    
    #                 if i == 10:
    #                     visualize_results(input, input, true_mask, pred_mask, epoch+1, text='phase_1')
                    
    #                 loss = fmd_v2.calculate_loss(pred_mask, true_mask)
    #                 total_loss_seg_val += loss.item()
            
    #         print(f"loss_seg_train: {total_loss_seg/len(train_real_loader):.4f}\t loss_seg_val: {total_loss_seg_val/len(test_real_loader):.4f}") 
                
    # phase 2: train with fake image
    epochs = config['model']['epoch']
    best_loss_seg = 1e6
    count = 0
    for epoch in tqdm(range(50)):
        torch.cuda.empty_cache()
        total_loss_seg = 0.0

        for (input, true_mask, real_image) in train_fake_loader:
            fmd_v2.set_input(input, true_mask, real_image)
            loss_seg = fmd_v2.optimize_parameters()
            
            # total loss
            total_loss_seg += loss_seg.item()
            
        # eval    
        # visualize some sample in test loader
        total_loss_seg_val = 0.0
        total_psnr = 0.0
        with torch.no_grad():
            for i, (input, true_mask, real_image) in enumerate(test_fake_loader):
                fmd_v2.set_input(input, true_mask, real_image)
                pred_mask, _, _, _, _, _, _ = fmd_v2()
                
                if ((epoch + 1) % 5 == 0 or epoch == 0) and i == 20:
                    visualize_results(input, real_image, true_mask, pred_mask, args.save_results, epoch+1, text='phase_2')
                
                # segmemtation loss
                loss = fmd_v2.calculate_loss(pred_mask, true_mask)
                total_loss_seg_val += loss.item()
                
                # psnr
                psnr = compute_psnr_batch(true_mask, pred_mask, device=device)
                total_psnr += psnr.item()
        
        print(f"loss_seg_train: {total_loss_seg/len(train_fake_loader):.4f}\t loss_seg_val: {total_loss_seg_val/len(test_fake_loader):.4f}\t \
            psnr: {total_psnr/len(test_fake_loader):.4f}")
        
        # Early Stopping
        if total_loss_seg_val < best_loss_seg:
            best_loss_seg = total_loss_seg_val
            count = 0
            # save checkpoint
            checkpoint_path = f'checkpoints/weights_attn_gate.pt'
            torch.save(fmd_v2.state_dict(), checkpoint_path)
                
        elif abs(total_loss_seg_val - best_loss_seg) < 0.001 or (total_loss_seg_val - best_loss_seg) > 0.01:
            count += 1
            if count == 5:
                break
        
if __name__ == '__main__':
    main()