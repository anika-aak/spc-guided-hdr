import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleFusion(nn.Module):
    """
    Fuse CMOS and SPC feature maps into a single 3-channel map.
    Steps:
    - Upsample SPC feat to CMOS feat spatial size.
    - Concat [CMOS, SPC_up] -> (B,6,H,W)
    - 1x1 conv -> (B,3,H,W)
    """
    def __init__(self):
        super().__init__()
        self.fuse_conv = nn.Conv2d(
            in_channels=6,
            out_channels=3,
            kernel_size=1
        )

    def forward(self, cmos_feat, spc_feat):
        # cmos_feat: (B,3,Hc,Wc)
        # spc_feat:  (B,3,Hs,Ws)
        B, _, Hc, Wc = cmos_feat.shape

        spc_up = F.interpolate(
            spc_feat,
            size=(Hc, Wc),
            mode="bilinear",
            align_corners=False
        )  # (B,3,Hc,Wc)

        fused6 = torch.cat([cmos_feat, spc_up], dim=1)  # (B,6,Hc,Wc)
        fused3 = self.fuse_conv(fused6)                 # (B,3,Hc,Wc)
        return fused3
