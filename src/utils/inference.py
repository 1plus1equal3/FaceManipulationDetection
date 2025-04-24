import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

import torch
import torch.nn as nn

from src.models.FMD_v2 import FMD_v2
from src.data.dataloader import get_dataloader
from src.utils.util import visualize_results

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument('--weight_path', type=str, default='', help='Path to your pretrained weight')
    parser.add_argument('--num_image', type=int, default=3, help='Number of batch image using to infer')
    
    args = parser.parse_args()
    return args

args = parse_argument()

def load_pretrained_weight(model, weight_path):
    state_dict = torch.load(weight_path)
    
    model_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
    torch.load_state_dict(model_dict, strict=False)
    

def main():
    model = FMD_v2().to(device)
    
    # load pretrained weight
    load_pretrained_weight(model, weight_path=args.weight_path)
    
    # infer data
    test_fake_loader = get_dataloader(mode='test', is_real=False)
    
    # inference
    model.eval()
    with torch.no_grad():
        for i, (fake_image, true_mask, real_image) in enumerate(test_fake_loader):
            model.set_input(fake_image, true_mask, real_image)
            pred_mask, _, _, _, _, _, _ = model()
            
            visualize_results(fake_image, real_image, true_mask, pred_mask, 100 + i, text='attn_gate')
            
            if i >= args.num_image:
                break
            
if __name__ == '__main__':
    main()