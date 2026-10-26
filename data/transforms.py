from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def augment_2d(x: torch.Tensor, rotate_deg: float = 10.0, scale: float = 0.05,
               brightness: float = 0.1, contrast: float = 0.1, hflip: bool = False) -> torch.Tensor:
    """Random small affine + intensity jitter on a [C,H,W] tensor."""
    c, h, w = x.shape
    angle = math.radians((torch.rand(1).item() * 2 - 1) * rotate_deg)
    s = 1.0 + (torch.rand(1).item() * 2 - 1) * scale
    cos, sin = math.cos(angle) / s, math.sin(angle) / s
    theta = torch.tensor([[cos, -sin, 0.0], [sin, cos, 0.0]], dtype=x.dtype).unsqueeze(0)
    grid = F.affine_grid(theta, (1, c, h, w), align_corners=False)
    x = F.grid_sample(x.unsqueeze(0), grid, padding_mode="border", align_corners=False)[0]
    mean = x.mean()
    x = (x - mean) * (1 + (torch.rand(1).item() * 2 - 1) * contrast) + mean
    x = x + (torch.rand(1).item() * 2 - 1) * brightness
    if hflip and torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=[-1])
    return x


def augment_3d(x: torch.Tensor, y: torch.Tensor, flip_prob: float = 0.5,
               noise_std: float = 0.0):
    """Random axis flips (applied identically to image [C,D,H,W] and mask [D,H,W])."""
    for axis in range(3):
        if torch.rand(1).item() < flip_prob:
            x = torch.flip(x, dims=[axis + 1])
            y = torch.flip(y, dims=[axis])
    if noise_std > 0:
        x = x + torch.randn_like(x) * noise_std
    return x, y
