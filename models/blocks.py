from __future__ import annotations

import torch.nn as nn


def make_norm(dim: int, norm: str, channels: int) -> nn.Module:
    if norm == "batch":
        return (nn.BatchNorm2d if dim == 2 else nn.BatchNorm3d)(channels)
    if norm == "instance":
        return (nn.InstanceNorm2d if dim == 2 else nn.InstanceNorm3d)(channels, affine=True)
    if norm == "group":
        groups = next(g for g in (8, 4, 2, 1) if channels % g == 0)
        return nn.GroupNorm(groups, channels)
    raise ValueError(f"unknown norm '{norm}'")


def conv_block(dim: int, cin: int, cout: int, stride: int = 1, norm: str = "batch",
               n_convs: int = 2) -> nn.Sequential:
    conv = nn.Conv2d if dim == 2 else nn.Conv3d
    layers = []
    for i in range(n_convs):
        layers += [
            conv(cin if i == 0 else cout, cout, 3, stride if i == 0 else 1, 1, bias=False),
            make_norm(dim, norm, cout),
            nn.ReLU(inplace=True),
        ]
    return nn.Sequential(*layers)
