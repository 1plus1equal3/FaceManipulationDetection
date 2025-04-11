import os
import sys
import json

sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.optim as optim

from src.data.dataloader import get_dataloader
from src.utils.util import *
from src.models.FMD import *
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

config = load_config('src/config.json')

def main():
    train_loader = get_dataloader(mode='train')
    test_loader = get_dataloader(mode='test')
    
    # define model
    fmd = FMD(device=device)
    
    epochs = config['model']['epoch']
    for epoch in range(epochs):
        total_loss_seg = 0.0
        total_loss_gen = 0.0
        total_loss_D_fake = 0.0
        total_loss_D_real = 0.0
        
        for (input, true_mask, input_is_real) in tqdm(train_loader):
            fmd.set_input(input, true_mask, input_is_real)
            loss_seg, loss_gen, loss_D_fake, loss_D_real = fmd.optimize_parameters()
            
            # total loss
            total_loss_seg += loss_seg
            total_loss_gen += loss_gen
            total_loss_D_fake += loss_D_fake
            total_loss_D_real += loss_D_real
            
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"loss_seg: {total_loss_seg/len(train_loader):.4f}, loss_gen: {total_loss_gen/len(train_loader):.4f}, loss_D_fake: {total_loss_D_fake/len(train_loader):.4f}, loss_D_real: {total_loss_D_real/len(train_loader):.4f}")
            
            # visualize some sample in test loader
            with torch.no_grad():
                for input, true_mask, input_is_real in test_loader:
                    fmd.set_input(input, true_mask, input_is_real)
                    rec_img, pred_mask, _, _, _, _, _ = fmd()
                    
                    visualize_results(input, rec_img, true_mask, pred_mask, epoch)
                    break
if __name__ == '__main__':
    main()