from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

from .blocks import conv_block
from .cag import GateCascade, PoolSize
from .physics import PhysicsPrior


class PhysioCAG2D(nn.Module):
    """2-D multi-label classifier: [Pi prior] -> CNN encoder -> CAG cascade -> GAP -> MLP."""

    def __init__(self, in_channels: int = 1, num_classes: int = 14, width: int = 32,
                 latent_dim: int = 128, guide_dims: Sequence[int] = (8, 4),
                 use_physics: bool = True, use_cag: bool = True, cag_pool: PoolSize = None,
                 attention: str = "spatial", norm: str = "batch", dropout: float = 0.3):
        super().__init__()
        self.physics = PhysicsPrior() if use_physics else None
        cin = in_channels * (2 if use_physics else 1)
        chans = [width, width * 2, width * 4, latent_dim]
        stages, prev = [], cin
        for c in chans:
            stages.append(conv_block(2, prev, c, stride=2, norm=norm, n_convs=1))
            prev = c
        self.encoder = nn.Sequential(*stages)
        self.cag = (GateCascade(latent_dim, guide_dims, pool_size=cag_pool, attention=attention)
                    if use_cag and len(guide_dims) > 0 else None)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(latent_dim, 256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor, guides: Optional[Sequence[torch.Tensor]] = None):
        if self.physics is not None:
            x = torch.cat([x, self.physics(x)], dim=1)
        feat = self.encoder(x)
        if self.cag is not None:
            feat = self.cag(feat, guides)
        return self.head(feat)
