import torchvision.transforms.functional as F
from PIL import Image
from .common import VIS_DIR

def _to_uint8_rgb(t3):
    # t3: (3,H,W) in [0,1]
    t = t3.clamp(0,1)
    t = (t*255.0).byte()
    return F.to_pil_image(t)

def _to_uint8_gray_as_rgb(t1):
    # t1: (1,H,W) in [0,1], tile to 3-ch for visualization
    t3 = t1.repeat(3,1,1).clamp(0,1)
    t3 = (t3*255.0).byte()
    return F.to_pil_image(t3)

def save_panel(cmos, spc, gt, recon, name):
    """
    All args are single examples on CPU:
    cmos:  (3,H,W)
    spc:   (1,H,W)
    gt:    (3,H,W)
    recon: (3,H,W)
    We create CMOS | SPC | GT | Recon side by side.
    """
    cmos_img  = _to_uint8_rgb(cmos)
    spc_img   = _to_uint8_gray_as_rgb(spc)
    gt_img    = _to_uint8_rgb(gt)
    recon_img = _to_uint8_rgb(recon)

    w, h = cmos_img.size
    panel = Image.new("RGB", (w*4, h))
    panel.paste(cmos_img,  (0,0))
    panel.paste(spc_img,   (w,0))
    panel.paste(gt_img,    (w*2,0))
    panel.paste(recon_img, (w*3,0))

    out_path = VIS_DIR / f"{name}_panel.png"
    panel.save(out_path)
    return out_path
