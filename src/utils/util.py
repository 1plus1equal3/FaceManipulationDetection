import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageChops, ImageEnhance
from io import BytesIO

def visualize_results(true_images, rec_images, true_masks, pred_masks,\
    d1s, d2s, d3s, d4s, d5s, d6s, save_folder, epoch, text='phase_2'):
    
    true_images = true_images.cpu().numpy()
    rec_images = rec_images.cpu().numpy()
    true_masks = true_masks.cpu().numpy()
    pred_masks = pred_masks.cpu().numpy()
    d1s = d1s.cpu().numpy()
    d2s = d2s.cpu().numpy()
    d3s = d3s.cpu().numpy()
    d4s = d4s.cpu().numpy()
    d5s = d5s.cpu().numpy()
    d6s = d6s.cpu().numpy()
    
    fig, axes = plt.subplots(32, 10, figsize=(20, 64))
    axes = axes.flatten()
    
    for i in range(32):
        true_image = np.transpose(true_images[i], (1, 2, 0))
        rec_image = np.transpose(rec_images[i], (1, 2, 0))
        # make sure that rec img in range(0, 1) instead (-1, 1)
        true_image = (true_image + 1) / 2
        # rec_image = (rec_image + 1) / 2
        
        true_mask = np.transpose(true_masks[i], (1, 2, 0))
        pred_mask = np.transpose(pred_masks[i], (1, 2, 0))
        d1 = np.transpose(d1s[i], (1, 2, 0))
        d2 = np.transpose(d2s[i], (1, 2, 0))
        d3 = np.transpose(d3s[i], (1, 2, 0))
        d4 = np.transpose(d4s[i], (1, 2, 0))
        d5 = np.transpose(d5s[i], (1, 2, 0))
        d6 = np.transpose(d6s[i], (1, 2, 0))
        
        axes[10*i].imshow(true_image)
        axes[10*i].set_title(f"Input image")
        axes[10*i].axis("off")
        
        axes[10*i+1].imshow(rec_image)
        axes[10*i+1].set_title(f"Ela image")
        axes[10*i+1].axis("off")
        
        axes[10*i+2].imshow(true_mask, cmap='jet')
        axes[10*i+2].set_title(f"True mask")
        axes[10*i+2].axis("off")
        
        axes[10*i+3].imshow(pred_mask, cmap='jet')
        axes[10*i+3].set_title(f"Predict mask")
        axes[10*i+3].axis("off")
        
        axes[10*i+4].imshow(d1, cmap='jet')
        axes[10*i+4].set_title(f"d1")
        axes[10*i+4].axis("off")
        
        axes[10*i+5].imshow(d2, cmap='jet')
        axes[10*i+5].set_title(f"d2")
        axes[10*i+5].axis("off")
        
        axes[10*i+6].imshow(d3, cmap='jet')
        axes[10*i+6].set_title(f"d3")
        axes[10*i+6].axis("off")
        
        axes[10*i+7].imshow(d4, cmap='jet')
        axes[10*i+7].set_title(f"d4")
        axes[10*i+7].axis("off")
        
        axes[10*i+8].imshow(d5, cmap='jet')
        axes[10*i+8].set_title(f"d5")
        axes[10*i+8].axis("off")
        
        axes[10*i+9].imshow(d6, cmap='jet')
        axes[10*i+9].set_title(f"d6")
        axes[10*i+9].axis("off")
    
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

# SSIM
def gaussian_window(window_size: int, sigma: float):
    gauss = torch.tensor([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    gauss /= gauss.sum()
    return gauss.unsqueeze(0) * gauss.unsqueeze(1)

def create_window(window_size: int, channel: int, device):
    window = gaussian_window(window_size, 1.5).to(device=device, dtype=torch.float32)
    window = window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim_batch(img1, img2, window_size=11, C1=0.01**2, C2=0.03**2):
    """
    Compute SSIM over a batch of images.
    
    img1, img2: tensors with shape (N, C, H, W), values in [0, 1]
    Returns average SSIM score over the batch.
    """
    assert img1.shape == img2.shape, "Input images must have the same shape"
    N, C, H, W = img1.size()
    device = img1.device
    img2 = img2.to(device)

    window = create_window(window_size, C, device)

    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=C)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=C)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=C) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean(dim=[1, 2, 3]).mean()  # mean SSIM over spatial, channel, and batch

def convert_to_ela_image(image_path, quality=90):
    image = Image.open(image_path).convert('RGB')

    # Save image to an in-memory buffer at specified JPEG quality
    buffer = BytesIO()
    image.save(buffer, 'JPEG', quality=quality)
    buffer.seek(0)

    # Load compressed image from the buffer
    compressed_image = Image.open(buffer)

    # Calculate ELA image (difference)
    ela_image = ImageChops.difference(image, compressed_image)

    # Enhance the difference
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0 / max_diff if max_diff != 0 else 1
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    return ela_image