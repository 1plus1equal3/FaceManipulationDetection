import json
import os
import sys
import wandb
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

    # wandb
    wandb.login(key='a6c7a7239821c37091bafec2b64870c6ca1aedfe')
    wandb.init(
        project="U2Net_MHSA_MHCA",
        config={
            "learning_rate": 0.0001,
            "batch_size": 32,
            "model": "U2Net with MHCA + MHSA",
            "loss_function": "L1 + CrossEntropy"
        }
    )
    
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
    
    best_loss_seg = 1e6
    count = 0
    for epoch in tqdm(range(config['model']['epoch'])):
        # set training mode
        fmd_v2.train()
        
        torch.cuda.empty_cache()
        total_loss_seg = 0.0
        total_loss_cls = 0.0
        true_preds = 0
        total = 0

        for (input, true_mask, ela, true_label) in train_combined_loader:
            fmd_v2.set_input(inputs=input, segment_labels=true_mask, ela=ela, cls_labels=true_label)
            
            # # training cls branch each 5 epochs
            # if (epoch + args.resume_epoch + 1) % 5 == 0 or epoch == 0: 
            #     loss_seg, loss_cls = fmd_v2.optimize_parameters(freeze_cls_branch=False)
            # else:
            #     loss_seg, loss_cls = fmd_v2.optimize_parameters(freeze_cls_branch=True)
            loss_seg, loss_cls = fmd_v2.optimize_parameters(freeze_cls_branch=False)

            # total loss
            total_loss_seg += loss_seg.item()
            total_loss_cls += loss_cls.item()
            
            true_pred, num_image = fmd_v2.get_num_true_pred_images()
            true_preds += true_pred
            total += num_image
            
        # eval
        fmd_v2.eval()
        # visualize some sample in test loader
        total_loss_seg_val = 0.0
        total_loss_cls_val = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        true_preds_val = 0
        total_val = 0
        psnr_bald = 0
        psnr_eyeglass = 0
        psnr_smile = 0
        with torch.no_grad():
            for i, (input, true_mask, ela, true_label) in enumerate(test_combined_loader):
                fmd_v2.set_input(inputs=input, segment_labels=true_mask, ela=ela, cls_labels=true_label)
                pred_mask, d1, d2, d3, d4, d5, pred_label = fmd_v2()

                # sigmoid
                pred_mask, d1, d2, d3, d4, d5 = torch.sigmoid(pred_mask), torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(d4), torch.sigmoid(d5)
                
                if ((epoch + 1) % 5 == 0 or epoch == 0) and i == 20:
                    visualize_results(input, ela, true_mask, pred_mask, d1, d2, d3, d4, d5, d5, args.save_results, epoch+1)
 
                # loss
                loss_seg, loss_cls = fmd_v2.backward_u2net_gan_v2(training=False)
                total_loss_seg_val += loss_seg.item()
                total_loss_cls_val += loss_cls.item()
                
                # accuracy
                true_pred, num_image = fmd_v2.get_num_true_pred_images()
                true_preds_val += true_pred
                total_val += num_image
                
                # psnr
                psnr = compute_psnr_batch(true_mask, pred_mask, device=device)
                total_psnr += psnr.item()
                
                # ssim
                ssim = ssim_batch(pred_mask, true_mask)
                total_ssim += ssim.item()

            if (epoch + args.resume_epoch + 1) % 5 == 0 or epoch == 0:
                # psnr for bald
                for (input, true_mask, ela, true_label) in bald_loader:
                    fmd_v2.set_input(inputs=input, segment_labels=true_mask, ela=ela, cls_labels=true_label)
                    pred_mask, d1, d2, d3, d4, d5, pred_label = fmd_v2()

                    # sigmoid
                    pred_mask, d1, d2, d3, d4, d5 = torch.sigmoid(pred_mask), torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(d4), torch.sigmoid(d5)

                    # psnr
                    psnr = compute_psnr_batch(true_mask, pred_mask, device=device)
                    psnr_bald += psnr.item()

                # psnr for eyeglass
                for (input, true_mask, ela, true_label) in eyeglass_loader:
                    fmd_v2.set_input(inputs=input, segment_labels=true_mask, ela=ela, cls_labels=true_label)
                    pred_mask, d1, d2, d3, d4, d5, pred_label = fmd_v2()

                    # sigmoid
                    pred_mask, d1, d2, d3, d4, d5 = torch.sigmoid(pred_mask), torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(d4), torch.sigmoid(d5)
                    
                    # psnr
                    psnr = compute_psnr_batch(true_mask, pred_mask, device=device)
                    psnr_eyeglass += psnr.item()

                # psnr for smile
                for (input, true_mask, ela, true_label) in smile_loader:
                    fmd_v2.set_input(inputs=input, segment_labels=true_mask, ela=ela, cls_labels=true_label)
                    pred_mask, d1, d2, d3, d4, d5, pred_label = fmd_v2()

                    # sigmoid
                    pred_mask, d1, d2, d3, d4, d5 = torch.sigmoid(pred_mask), torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(d4), torch.sigmoid(d5)
                    
                    # psnr
                    psnr = compute_psnr_batch(true_mask, pred_mask, device=device)
                    psnr_smile += psnr.item()
        
        print(
            f"\nEpoch: {epoch + args.resume_epoch + 1}\n"
            f"loss_seg_train: {total_loss_seg/len(train_combined_loader):.4f}\n"
            f"loss_seg_val: {total_loss_seg_val/len(test_combined_loader):.4f}\n"
            f"loss_cls_train: {total_loss_cls/len(train_combined_loader):.4f}\n"
            f"loss_cls_val: {total_loss_cls_val/len(test_combined_loader):.4f}\n"
            f"acc_train: {true_preds/total:.4f}\n"
            f"acc_val: {true_preds_val/total_val:.4f}\n"
            f"total_psnr: {total_psnr/len(test_combined_loader):.4f}\n"
            f"ssim: {total_ssim/len(test_combined_loader):.4f}\n"
            f"bald_psnr: {psnr_bald/len(bald_loader):.4f}\n"
            f"eyeglass_psnr: {psnr_eyeglass/len(eyeglass_loader):.4f}\n"
            f"smile_psnr: {psnr_smile/len(smile_loader):.4f}\n"
        )
        
        wandb.log({
            "train_seg_loss": total_loss_seg/len(train_combined_loader),
            "test_seg_loss": total_loss_seg_val/len(test_combined_loader),
            "train_acc": true_preds/total,
            "test_acc": true_preds_val/total_val,
            "psnr": total_psnr/len(test_combined_loader),
            "ssim": total_ssim/len(test_combined_loader),
            "bald_psnr": psnr_bald/len(bald_loader),
            "eyeglass_psnr": psnr_eyeglass/len(eyeglass_loader),
            "smile_psnr": psnr_smile/len(smile_loader),
        })

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