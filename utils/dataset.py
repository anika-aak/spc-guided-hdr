import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
from .common import DATA_ROOT

class HDRPairDataset(Dataset):
    """
    Task A Dataset:
    Loads CMOS and SPC images with identical filenames.
    No resizing, no ground truth, no decoder.
    """

    def __init__(self):
        # --- exact directories ---
        self.cmos_dir = (
            DATA_ROOT
            / "Training_1024x512"
            / "CMOS_sat_1024x512"
            / "dynamic_exposures_train_png_1024x512"
        )
        self.spc_dir = (
            DATA_ROOT
            / "Training_1024x512"
            / "SPC_512x256_train_png"
        )

        # --- load matching file pairs ---
        self.items = self._pair_lists()
        self.to_tensor_rgb = T.ToTensor()
        self.to_tensor_gray = T.ToTensor()

    def _pair_lists(self):
        """Pair CMOS and SPC images by exact same stem name."""
        cmos_files = [f for f in os.listdir(self.cmos_dir) if f.lower().endswith(".png")]
        spc_files  = [f for f in os.listdir(self.spc_dir) if f.lower().endswith(".png")]

        cmos_map = {Path(f).stem: self.cmos_dir / f for f in cmos_files}
        spc_map  = {Path(f).stem: self.spc_dir / f for f in spc_files}

        common = sorted(set(cmos_map.keys()) & set(spc_map.keys()))
        if not common:
            raise RuntimeError(
                f"No matched pairs found.\nCMOS dir: {self.cmos_dir}\nSPC dir: {self.spc_dir}"
            )

        return [{"name": k, "cmos_path": cmos_map[k], "spc_path": spc_map[k]} for k in common]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        entry = self.items[idx]
        cmos_img = Image.open(entry["cmos_path"]).convert("RGB")
        spc_img  = Image.open(entry["spc_path"]).convert("L")

        return {
            "name": entry["name"],
            "cmos": self.to_tensor_rgb(cmos_img),
            "spc":  self.to_tensor_gray(spc_img)
        }
