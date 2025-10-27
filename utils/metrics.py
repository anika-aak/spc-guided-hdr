import torch
import torch.nn.functional as F
import math

def mse_loss(pred, target):
    return F.mse_loss(pred, target)

def psnr(pred, target, max_val=1.0):
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return 99.0
    return 10.0 * math.log10((max_val ** 2) / mse)

def ssim(pred, target, C1=0.01**2, C2=0.03**2):
    """
    pred, target: (B,3,H,W) in [0,1]
    Very lightweight SSIM, patch-based via avg pooling.
    """
    mu_x = F.avg_pool2d(pred, 3, 1, 1)
    mu_y = F.avg_pool2d(target, 3, 1, 1)

    sigma_x  = F.avg_pool2d(pred * pred, 3, 1, 1) - mu_x * mu_x
    sigma_y  = F.avg_pool2d(target * target, 3, 1, 1) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred * target, 3, 1, 1) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
        (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
    )
    return ssim_map.mean().item()
