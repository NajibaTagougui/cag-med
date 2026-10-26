from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceCELoss(nn.Module):
    """Cross-entropy + soft Dice for binary segmentation with 2-channel logits."""

    def __init__(self, ce_weight: float = 1.0, dice_weight: float = 1.0, eps: float = 1e-5):
        super().__init__()
        self.ce_weight, self.dice_weight, self.eps = ce_weight, dice_weight, eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        ce = F.cross_entropy(logits, target.long())
        prob = torch.softmax(logits, dim=1)[:, 1].flatten(1)
        tgt = (target > 0).float().flatten(1)
        inter = (prob * tgt).sum(1)
        dice = (2 * inter + self.eps) / (prob.sum(1) + tgt.sum(1) + self.eps)
        return self.ce_weight * ce + self.dice_weight * (1 - dice).mean()


def build_loss(cfg: dict, pos_weight: Optional[torch.Tensor] = None) -> nn.Module:
    if cfg["task"] == "classification":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if cfg["task"] == "segmentation":
        return DiceCELoss()
    raise ValueError(f"unknown task '{cfg['task']}'")
