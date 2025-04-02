import torch
import numpy as np
import matplotlib.pyplot as plt

def visualize_results(images: torch.Tensor, masks: torch.Tensor):
    images = images.cpu().numpy()
    masks = masks.cpu().numpy()
    
    fig, axes = plt.subplots(8, 4, figsize=(12, 24))
    axes = axes.flatten()
    
    for i in range(16):
        img = np.transpose(images[i], (1, 2, 0))
        mask = np.transpose(masks[i], (1, 2, 0))
        
        axes[2*i].imshow(img)
        axes[2*i].set_title(f"Real Mask")
        axes[2*i].axis("off")
        
        axes[2*i + 1].imshow(mask, cmap='gray')
        axes[2*i + 1].set_title(f"Predict Mask")
        axes[2*i + 1].axis("off")
    
    plt.tight_layout()
    plt.show()