import os
from pathlib import Path
import torch

# Root where the 6 data folders already live
DATA_ROOT = Path("Low_Resolution_Training_1024x512")

# Output root for all artifacts we’ll show in review
OUT_ROOT = Path("outputs")
OUT_ROOT.mkdir(exist_ok=True)

# Subfolders for saving intermediate images
FEAT_SPC_DIR = OUT_ROOT / "feat_spc"
FEAT_CMOS_DIR = OUT_ROOT / "feat_cmos"
RECON_DIR = OUT_ROOT / "reconstructions"
VIS_DIR = OUT_ROOT / "visuals_for_review"

for d in [FEAT_SPC_DIR, FEAT_CMOS_DIR, RECON_DIR, VIS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {DEVICE}")
