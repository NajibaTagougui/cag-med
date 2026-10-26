"""Cross-Attention Gating (CAG): condition a spatial feature map on a continuous guide vector.

    T~ = T * sigmoid(Omega),   Omega = A V^T

Queries come from the N spatial tokens of the feature map; key and value come from a single
guide token. Two attention variants are provided:

* ``attention="spatial"`` (default): softmax over the N spatial positions, rescaled by N so
  the mean attention weight is 1. The gate varies across space.
* ``attention="legacy"``: softmax over the singleton key axis, as in the original research
  code. This is identically 1, so the gate is spatially constant (a channel-wise gate from
  the guide). Kept only to reproduce earlier results.

With ``guide == 0`` the value is exactly 0 (no bias on the guide projection or W_V), so
sigmoid(0) = 0.5 and the gate halves the features.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

PoolSize = Optional[Union[int, Sequence[int]]]


def _pool_target(pool_size: PoolSize, spatial: Tuple[int, ...]) -> Optional[Tuple[int, ...]]:
    """Return the pooled size, or None if pooling would not shrink the token grid."""
    if pool_size is None:
        return None
    if isinstance(pool_size, int):
        target = (pool_size,) * len(spatial)
    else:
        target = tuple(pool_size)
        if len(target) != len(spatial):
            raise ValueError(f"pool_size {target} does not match spatial dims {spatial}")
    target = tuple(min(t, s) for t, s in zip(target, spatial))
    return None if target == tuple(spatial) else target


class CrossAttentionGate(nn.Module):
    def __init__(self, channels: int, guide_dim: int, pool_size: PoolSize = None,
                 attention: str = "spatial"):
        super().__init__()
        if attention not in ("spatial", "legacy"):
            raise ValueError("attention must be 'spatial' or 'legacy'")
        self.channels = channels
        self.guide_dim = guide_dim
        self.pool_size = pool_size
        self.attention = attention
        self.guide_proj = nn.Linear(guide_dim, channels, bias=False)
        self.w_q = nn.Linear(channels, channels)
        self.w_k = nn.Linear(channels, channels)
        self.w_v = nn.Linear(channels, channels, bias=False)
        self.scale = 1.0 / math.sqrt(channels)
        # Optional introspection (used by physiocag.visualize)
        self.store_gate = False
        self.last_gate: Optional[torch.Tensor] = None

    def forward(self, feat: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        batch, channels = feat.shape[:2]
        spatial = tuple(feat.shape[2:])
        if channels != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {channels}")

        x = feat
        target = _pool_target(self.pool_size, spatial)
        if target is not None:
            pool = F.adaptive_avg_pool2d if len(spatial) == 2 else F.adaptive_avg_pool3d
            x = pool(x, target)
        coarse = tuple(x.shape[2:])

        tokens = x.flatten(2).transpose(1, 2)                   # B, N', C
        n_tokens = tokens.shape[1]
        g = F.relu(self.guide_proj(guide))                      # B, C
        q = self.w_q(tokens)                                    # B, N', C
        k = self.w_k(g).unsqueeze(1)                            # B, 1, C
        v = self.w_v(g).unsqueeze(1)                            # B, 1, C

        logits = (q * k).sum(-1, keepdim=True) * self.scale     # B, N', 1
        if self.attention == "spatial":
            attn = torch.softmax(logits, dim=1) * n_tokens
        else:
            attn = torch.softmax(logits, dim=-1)                # identically 1

        omega = (attn * v).transpose(1, 2).reshape(batch, channels, *coarse)
        if coarse != spatial:
            mode = "bilinear" if len(spatial) == 2 else "trilinear"
            omega = F.interpolate(omega, size=spatial, mode=mode, align_corners=False)
        gate = torch.sigmoid(omega)
        if self.store_gate:
            self.last_gate = gate.detach()
        return feat * gate


class GateCascade(nn.Module):
    """Sequential CAG stages, one per guide stream (e.g. omics first, vitals second)."""

    def __init__(self, channels: int, guide_dims: Sequence[int], pool_size: PoolSize = None,
                 attention: str = "spatial"):
        super().__init__()
        self.gates = nn.ModuleList(
            CrossAttentionGate(channels, d, pool_size=pool_size, attention=attention)
            for d in guide_dims
        )

    def forward(self, feat: torch.Tensor, guides: Optional[Sequence[torch.Tensor]] = None):
        if guides is not None and len(guides) != len(self.gates):
            raise ValueError(f"expected {len(self.gates)} guide tensors, got {len(guides)}")
        for i, gate in enumerate(self.gates):
            g = guides[i] if guides is not None else None
            if g is None:
                g = feat.new_zeros(feat.shape[0], gate.guide_dim)
            feat = gate(feat, g.to(feat.dtype))
        return feat
