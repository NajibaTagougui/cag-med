from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import conv_block
from .cag import GateCascade, PoolSize
from .physics import PhysicsPrior


class PhysioCAG3D(nn.Module):
    """3-D U-Net style segmenter with the CAG cascade at the bottleneck.

    Input spatial sizes should be divisible by 8.
    """

    def __init__(self, in_channels: int = 4, num_classes: int = 2, base: int = 16,
                 guide_dims: Sequence[int] = (8, 4), use_physics: bool = True,
                 use_cag: bool = True, cag_pool: PoolSize = None, attention: str = "spatial",
                 norm: str = "instance"):
        super().__init__()
        self.physics = PhysicsPrior() if use_physics else None
        cin = in_channels * (2 if use_physics else 1)
        c = [base, base * 2, base * 4, base * 8]
        self.enc0 = conv_block(3, cin, c[0], 1, norm)
        self.enc1 = conv_block(3, c[0], c[1], 2, norm)
        self.enc2 = conv_block(3, c[1], c[2], 2, norm)
        self.enc3 = conv_block(3, c[2], c[3], 2, norm)
        self.cag = (GateCascade(c[3], guide_dims, pool_size=cag_pool, attention=attention)
                    if use_cag and len(guide_dims) > 0 else None)
        self.up2 = nn.ConvTranspose3d(c[3], c[2], 2, 2)
        self.dec2 = conv_block(3, c[2] * 2, c[2], 1, norm)
        self.up1 = nn.ConvTranspose3d(c[2], c[1], 2, 2)
        self.dec1 = conv_block(3, c[1] * 2, c[1], 1, norm)
        self.up0 = nn.ConvTranspose3d(c[1], c[0], 2, 2)
        self.dec0 = conv_block(3, c[0] * 2, c[0], 1, norm)
        self.out = nn.Conv3d(c[0], num_classes, 1)

    @staticmethod
    def _match(up: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if up.shape[2:] != skip.shape[2:]:
            up = F.interpolate(up, size=skip.shape[2:], mode="trilinear", align_corners=False)
        return up

    def forward(self, x: torch.Tensor, guides: Optional[Sequence[torch.Tensor]] = None):
        if self.physics is not None:
            x = torch.cat([x, self.physics(x)], dim=1)
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        if self.cag is not None:
            e3 = self.cag(e3, guides)
        d2 = self.dec2(torch.cat([self._match(self.up2(e3), e2), e2], 1))
        d1 = self.dec1(torch.cat([self._match(self.up1(d2), e1), e1], 1))
        d0 = self.dec0(torch.cat([self._match(self.up0(d1), e0), e0], 1))
        return self.out(d0)
