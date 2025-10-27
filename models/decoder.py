import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallDecoder(nn.Module):
    """
    Decoder that takes BOTH:
    - fused_feat (3ch guidance from CMOS+SPC)
    - raw CMOS image (3ch with spatial detail)
    Concatenate -> 6 channels, then reconstruct HDR-ish RGB.
    """

    def __init__(self, in_channels=6, mid=64):
        super().__init__()

        # a slightly deeper conv stack to give it some capacity
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid, mid, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid, mid, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.out = nn.Sequential(
            nn.Conv2d(mid, 32, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 1),
            nn.Sigmoid(),       # keep output in [0,1] so loss vs GT works
        )

    def forward(self, x):
        x = self.enc(x)
        x = self.out(x)
        return x
