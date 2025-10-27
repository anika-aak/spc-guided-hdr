import torch
import torchvision.transforms.functional as F
from PIL import Image
from .common import FEAT_SPC_DIR, FEAT_CMOS_DIR

def _tensor_to_pngimg(t3):
    """
    t3: (3,H,W) float tensor, any scale.
    We min-max normalize to [0,255] so it’s visible as an image.
    """
    with torch.no_grad():
        t = t3.clone()
        t = t - t.amin(dim=[1,2], keepdim=True)
        denom = t.amax(dim=[1,2], keepdim=True).clamp(min=1e-6)
        t = t / denom
        t = (t * 255.0).clamp(0,255).byte()  # (3,H,W)
        pil = F.to_pil_image(t)
        return pil

def save_feature_map(feat_tensor, name, mode="spc"):
    """
    feat_tensor: (B,3,H,W) torch float
    name: string (filename stem)
    mode: "spc" or "cmos"
    """
    pil_img = _tensor_to_pngimg(feat_tensor.cpu()[0])  # take first in batch
    if mode == "spc":
        out_path = FEAT_SPC_DIR / f"{name}_feat.png"
    else:
        out_path = FEAT_CMOS_DIR / f"{name}_feat.png"
    pil_img.save(out_path)
    return out_path

def tensor_to_pngimg(t3_cpu):
    """
    Convenience for reconstructed HDR saving later.
    t3_cpu: (3,H,W) on CPU, [0,1] expected. We'll just clamp and scale.
    """
    t = t3_cpu.clamp(0,1)
    t = (t*255.0).byte()
    pil = F.to_pil_image(t)
    return pil
