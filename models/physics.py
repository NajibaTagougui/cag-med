"""Parameter-free projection operator Pi.

``descriptors`` returns the mean projections along each spatial axis (row/column means for
2-D; axial/coronal/sagittal means for 3-D). ``forward`` back-projects them into a
full-resolution prior map with the same shape as the input, which can be concatenated to
the input as extra channels. The operator itself has no trainable parameters.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class PhysicsPrior(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    @staticmethod
    def descriptors(x: torch.Tensor) -> List[torch.Tensor]:
        if x.dim() not in (4, 5):
            raise ValueError("expected [B,C,H,W] or [B,C,D,H,W]")
        spatial_axes = range(2, x.dim())
        return [x.mean(dim=a, keepdim=True) for a in spatial_axes]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projections = self.descriptors(x)
        prior = projections[0]
        for p in projections[1:]:
            prior = prior + p                                   # broadcasts to full shape
        prior = (prior / len(projections)).expand_as(x)
        dims = tuple(range(2, x.dim()))
        mean = prior.mean(dim=dims, keepdim=True)
        std = prior.std(dim=dims, keepdim=True)
        return (prior - mean) / (std + self.eps)
