# SPC-Guided HDR Reconstruction

This repository implements a compact two-stage HDR reconstruction pipeline using **SPC (Single-Photon Counting)** and **CMOS saturated images**.

---

##  Project Overview

### **Task A – Feature Extraction**
- Extract per-sensor feature maps from:
  - **CMOS saturated RGB images**
  - **SPC grayscale images**
- Feature extraction modules:
  - `HighResCMOSEncoder` → preserves spatial detail (U-Net-like)
  - `DilatedConvEncoder` → captures sparse photon features
- Produces 3-channel feature maps for both sensors and a fused feature representation.

### **Task B – Reconstruction**
- Fuse CMOS + SPC features using `SimpleFusion`
- Decode to HDR-like RGB using `SmallDecoder`
- Evaluate reconstructed output using:
  - **MSE (Mean Squared Error)**
  - **PSNR (Peak Signal-to-Noise Ratio)**
  - **SSIM (Structural Similarity Index)**
- Saves intermediate and reconstructed images for visual comparison.

---

##  Folder Structure

SPC Guided HDR/
│
├── models/
│ ├── feature_extractors.py # DilatedConvEncoder, HighResCMOSEncoder
│ ├── decoder.py # SmallDecoder (6-channel input)
│ └── fusion.py # SimpleFusion block
│
├── utils/
│ ├── dataset.py # Loads CMOS + SPC + GT images
│ ├── metrics.py # PSNR, SSIM, MSE utilities
│ ├── save_intermediate.py # Image saving helpers
│ ├── viz.py # Side-by-side visualization panels
│ └── common.py # Paths, DEVICE configuration
│
├── taskA_feature_extract.ipynb # Generates feature maps (Task A)
├── taskB_decode_eval.ipynb # Trains decoder + evaluates (Task B)
│
├── requirements.txt # Python dependencies
├── .gitignore
└── Low_Resolution_Training_1024x512/ # Dataset (not uploaded)
├── Training_1024x512/
│ ├── CMOS_sat_1024x512/
│ │ └── dynamic_exposures_train_png_1024x512/
│ ├── SPC_512x256_train_png/
│ └── ...
└── GT_HDR_1024X512_train_png/

Task A – Feature Extraction

Open taskA_feature_extract.ipynb and run all cells.
This generates:

individual CMOS & SPC feature maps

optionally fused maps

outputs stored under outputs/taskA_outputs/

Task B – Decoder & Evaluation

Open taskB_decode_eval.ipynb and run all cells.
This:

trains the decoder (~600 epochs default)

reconstructs HDR images

computes MSE / PSNR / SSIM

saves results to outputs/recon/
---

##  Installation

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/spc-guided-hdr.git
cd spc-guided-hdr

pip install -r requirements.txt
