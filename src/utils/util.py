import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def visualize_results(true_images, rec_images, true_masks, pred_masks, save_folder, epoch, text='phase_1'):
    true_images = true_images.cpu().numpy()
    rec_images = rec_images.cpu().numpy()
    true_masks = true_masks.cpu().numpy()
    pred_masks = pred_masks.cpu().numpy()
    
    fig, axes = plt.subplots(32, 4, figsize=(8, 64))
    axes = axes.flatten()
    
    for i in range(32):
        true_image = np.transpose(true_images[i], (1, 2, 0))
        rec_image = np.transpose(rec_images[i], (1, 2, 0))
        # make sure that rec img in range(0, 1) instead (-1, 1)
        true_image = (true_image + 1) / 2
        rec_image = (rec_image + 1) / 2
        
        true_mask = np.transpose(true_masks[i], (1, 2, 0))
        pred_mask = np.transpose(pred_masks[i], (1, 2, 0))
        
        axes[4*i].imshow(true_image)
        axes[4*i].set_title(f"Input image")
        axes[4*i].axis("off")
        
        axes[4*i+1].imshow(rec_image)
        axes[4*i+1].set_title(f"Rec image")
        axes[4*i+1].axis("off")
        
        axes[4*i+2].imshow(true_mask, cmap='jet')
        axes[4*i+2].set_title(f"True mask")
        axes[4*i+2].axis("off")
        
        axes[4*i+3].imshow(pred_mask, cmap='jet')
        axes[4*i+3].set_title(f"Predict mask")
        axes[4*i+3].axis("off")
    
    plt.tight_layout()
    plt.savefig(f'{save_folder}/{text}_{epoch}_v2.png')
    
# PSNR calculation function for a batch
def compute_psnr_batch(original, reconstructed, device='cpu', max_pixel_value=1.0, epsilon=1e-10):
    """
    Calculate PSNR for a batch of images.
    Args:
        original: Tensor of shape (B, C, H, W), original images
        reconstructed: Tensor of shape (B, C, H, W), reconstructed images
        max_pixel_value: Maximum pixel value (1.0 for normalized, 255.0 for 8-bit)
        epsilon: Small value to avoid division by zero
    Returns:
        psnr: Tensor of shape (B,), PSNR for each image in the batch
    """
    original = original.to(device)
    reconstructed = reconstructed.to(device)
    
    mse = F.mse_loss(original, reconstructed, reduction='none').mean(dim=(1, 2, 3))
    psnr = 10 * torch.log10((max_pixel_value ** 2) / (mse + epsilon))
    psnr_avg = torch.mean(psnr)
    return psnr_avg