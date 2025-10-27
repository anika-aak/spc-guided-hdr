import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
class DilatedConvEncoder(nn.Module):
    """
    Dilation-based feature extractor.
    Input: N x C x H x W  (C=1 for SPC or 3 for CMOS)
    Output: N x 3 x H x W (3-channel feature map)
    """
    def __init__(self, in_channels=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=4, dilation=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=8, dilation=8),
            nn.ReLU(inplace=True),
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(64, 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.out_conv(x)  # -> (B,3,H,W)
        return x


class PretrainedBackboneEncoder(nn.Module):
    """
    ResNet18 backbone (ImageNet weights) → 3ch feature map,
    then upsample back to input spatial size.

    This is mainly for CMOS (3ch). For SPC (1ch), we just repeat the channel.
    """
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
        )

        # freeze
        for p in self.stem.parameters():
            p.requires_grad = False

        # project ResNet channels (256) down to 3
        self.proj = nn.Conv2d(256, 3, kernel_size=1)

    def forward(self, x):
        # if grayscale (1ch) tile to 3ch
        if x.shape[1] == 1:
            x = x.repeat(1,3,1,1)
        feats = self.stem(x)          # (B,256,H/16,W/16 approx)
        feats3 = self.proj(feats)     # (B,3,H/16,W/16)
        up = torch.nn.functional.interpolate(
            feats3, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        return up  # (B,3,H,W)



class HighResCMOSEncoder(nn.Module):
    """
    CMOS feature extractor that preserves spatial detail.
    Compared to ResNet18:
    - WAY less aggressive downsampling
    - Uses shallow UNet-ish skip connections
    - Returns 3-channel feature map at full CMOS resolution

    Input:  (B,3,H,W)  ~ (3,512,1024)
    Output: (B,3,H,W)
    """

    def __init__(self, base_channels=32):
        super().__init__()

        # Encoder path (light downsample to 1/4 res)
        # stage1: keep full res
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # stage2: downsample x2
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels*2, 3, stride=2, padding=1),  # /2
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*2, base_channels*2, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # stage3: downsample x2 again (total /4)
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels*4, 3, stride=2, padding=1),  # /4
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*4, base_channels*4, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Bottleneck convs (dilated for larger receptive field without more pooling)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels*4, base_channels*4, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*4, base_channels*4, 3, padding=4, dilation=4),
            nn.ReLU(inplace=True),
        )

        # Decoder / upsample path with skip connections
        # up from /4 -> /2
        self.up2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, 4, stride=2, padding=1)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_channels*2 + base_channels*2, base_channels*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*2, base_channels*2, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # up from /2 -> /1
        self.up1 = nn.ConvTranspose2d(base_channels*2, base_channels, 4, stride=2, padding=1)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels + base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # final projection to 3 channels
        self.out_conv = nn.Conv2d(base_channels, 3, 1)

    def forward(self, x):
        """
        x: (B,3,H,W)
        """
        # encoder
        e1 = self.enc1(x)         # (B,C,H,W)
        e2 = self.enc2(e1)        # (B,2C,H/2,W/2)
        e3 = self.enc3(e2)        # (B,4C,H/4,W/4)

        b = self.bottleneck(e3)   # (B,4C,H/4,W/4)

        # decoder with skips
        # upsample /4 -> /2
        u2 = self.up2(b)          # (B,2C,H/2,W/2)
        d2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(d2)        # (B,2C,H/2,W/2)

        # upsample /2 -> /1
        u1 = self.up1(d2)         # (B,C,H,W)
        d1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(d1)        # (B,C,H,W)

        out = self.out_conv(d1)   # (B,3,H,W)
        return out
