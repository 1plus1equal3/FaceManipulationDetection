import torch
import numpy as np
import matplotlib.pyplot as plt

def visualize_results(true_images, rec_images, true_masks, pred_masks):
    true_images = true_images.cpu().numpy()
    rec_images = rec_images.cpu().numpy()
    true_masks = true_masks.cpu().numpy()
    pred_masks = pred_masks.cpu().numpy()
    
    fig, axes = plt.subplots(16, 4, figsize=(12, 24))
    axes = axes.flatten()
    
    for i in range(16):
        true_image = np.transpose(true_images[i], (1, 2, 0))
        rec_image = np.transpose(rec_images[i], (1, 2, 0))
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
        
        axes[4*i+3].imshow(pred_mask)
        axes[4*i+3].set_title(f"Predict mask")
        axes[4*i+3].axis("off")
    
    plt.tight_layout()
    plt.show()