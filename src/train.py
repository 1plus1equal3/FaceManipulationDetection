import json
import os
import sys
import argparse
import random
import numpy as np

sys.path.append(os.getcwd())

import torch
from tqdm import tqdm

from src.data.dataloader import get_dataloader
from src.models.FMD_v2 import *
from src.utils.util import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

config = load_config('src/config.json')

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)

def main():
    # set random seed
    set_random_seed(42)

    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='', help='path to your dataset')
    parser.add_argument('--weights', type=str, default=None, help='path to your pretrained weight')
    parser.add_argument('--resume_epoch', type=int, default=0, help='resume epoch if using pretrained')
    parser.add_argument('--save_results', type=str, default='', help='path to save results')
    
    args = parser.parse_args()
    
    # define model
    fmd_v2 = FMD_v2(device=device)
    
    # load pre-trained weight
    if args.weights:
        model_dict = torch.load(args.weights, weights_only=True)
        
        # load weight
        pretrain_dict = {k: v for k, v in model_dict.items() if k in fmd_v2.state_dict()}
        fmd_v2.load_state_dict(pretrain_dict, strict=False)
    
    # dataloader
    train_combined_loader = get_dataloader(dataset_path=args.dataset_path, mode='train', type='combined')
    test_combined_loader = get_dataloader(dataset_path=args.dataset_path, mode='test', type='combined')
    bald_loader = get_dataloader(dataset_path=args.dataset_path, attribute='bald')
    eyeglass_loader = get_dataloader(dataset_path=args.dataset_path, attribute='eyeglass')
    smile_loader = get_dataloader(dataset_path=args.dataset_path, attribute='smile')
    
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
    best_loss_seg = 1e6
    count = 0
    for epoch in tqdm(range(config['model']['epoch'])):
        # set training mode
        fmd_v2.train()
        
        torch.cuda.empty_cache()
        total_loss_seg = 0.0

        for (input, true_mask, _, _) in train_combined_loader:
            fmd_v2.set_input(inputs=input, labels=true_mask)
            
            loss_seg = fmd_v2.optimize_parameters()
            
            # total loss
            total_loss_seg += loss_seg.item()
            
        # eval
        fmd_v2.eval()
        # visualize some sample in test loader
        total_loss_seg_val = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        psnr_bald = 0
        psnr_eyeglass = 0
        psnr_smile = 0
        with torch.no_grad():
            for i, (input, true_mask, _, _) in enumerate(test_combined_loader):
                fmd_v2.set_input(inputs=input, labels=true_mask)
                pred_mask, d1, d2, d3, d4, d5, d6 = fmd_v2()
                
                if ((epoch + 1) % 5 == 0 or epoch == 0) and i == 20:
                    visualize_results(input, input, true_mask, pred_mask, d1, d2, d3, d4, d5, d6, args.save_results, epoch+1)
 
                # loss
                loss_seg = fmd_v2.backward_u2net_gan_v2(training=False)
                total_loss_seg_val += loss_seg.item()

                
                # psnr
                psnr = compute_psnr_batch(true_mask, pred_mask, device=device)
                total_psnr += psnr.item()
                
                # ssim
                ssim = ssim_batch(pred_mask, true_mask)
                total_ssim += ssim.item()

            if (epoch + args.resume_epoch + 1) % 5 == 0 or epoch == 0:
                # psnr for bald
                for (input, true_mask, _, _) in bald_loader:
                    fmd_v2.set_input(inputs=input, labels=true_mask)
                    pred_mask, d1, d2, d3, d4, d5, d6 = fmd_v2()

                    # psnr
                    psnr = compute_psnr_batch(true_mask, pred_mask, device=device)
                    psnr_bald += psnr.item()

                # psnr for eyeglass
                for (input, true_mask, _, _) in eyeglass_loader:
                    fmd_v2.set_input(inputs=input, labels=true_mask)
                    pred_mask, d1, d2, d3, d4, d5, d6 = fmd_v2()

                    # psnr
                    psnr = compute_psnr_batch(true_mask, pred_mask, device=device)
                    psnr_eyeglass += psnr.item()

                # psnr for smile
                for (input, true_mask, _, _) in smile_loader:
                    fmd_v2.set_input(inputs=input, labels=true_mask)
                    pred_mask, d1, d2, d3, d4, d5, d6 = fmd_v2()

                    # psnr
                    psnr = compute_psnr_batch(true_mask, pred_mask, device=device)
                    psnr_smile += psnr.item()
        
        print(
            f"\nEpoch: {epoch + args.resume_epoch + 1}\n"
            f"loss_seg_train: {total_loss_seg/len(train_combined_loader):.4f}\n"
            f"loss_seg_val: {total_loss_seg_val/len(test_combined_loader):.4f}\n"
            f"total_psnr: {total_psnr/len(test_combined_loader):.4f}\n"
            f"ssim: {total_ssim/len(test_combined_loader):.4f}\n"
            f"bald_psnr: {psnr_bald/len(bald_loader):.4f}\n"
            f"eyeglass_psnr: {psnr_eyeglass/len(eyeglass_loader):.4f}\n"
            f"smile_psnr: {psnr_smile/len(smile_loader):.4f}\n"
        )
        
        # Early Stopping
        if total_loss_seg_val < best_loss_seg:
            best_loss_seg = total_loss_seg_val
            count = 0
            checkpoint_path = f'checkpoints/weights_attn_gate.pt'
            torch.save(fmd_v2.state_dict(), checkpoint_path)
        else:
            count += 1
            if count >= 5:
                print(f"Early Stopping after {epoch + args.resume_epoch + 1} epochs\n")
                break
        
if __name__ == '__main__':
    main()