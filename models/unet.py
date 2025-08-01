
import torch
import torch.nn as nn
from core.base_model import BaseVSModel
from core.registry import register

def conv_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1),
        nn.InstanceNorm2d(out_c),
        nn.ReLU(True)
    )

@register
class UNetVS(BaseVSModel):
    """Very small UNet for image→image virtual staining."""
    def __init__(self, hparams):
        super().__init__(hparams)
        self.enc1 = conv_block(3, 32)
        self.enc2 = conv_block(32, 64)
        self.enc3 = conv_block(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.dec2 = conv_block(128+64, 64)
        self.dec1 = conv_block(64+32, 32)
        self.final = nn.Conv2d(32, 3, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = torch.cat([nn.functional.interpolate(e3, scale_factor=2, mode='bilinear', align_corners=False), e2], 1)
        d2 = self.dec2(d2)
        d1 = torch.cat([nn.functional.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False), e1], 1)
        d1 = self.dec1(d1)
        return torch.tanh(self.final(d1))
