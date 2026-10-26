from __future__ import annotations

from typing import Any, Dict, Tuple

import torch.nn as nn

from .physiocag2d import PhysioCAG2D
from .physiocag3d import PhysioCAG3D


def build_model(cfg: Dict[str, Any]) -> nn.Module:
    m = dict(cfg["model"])
    name = m.pop("name")
    data = cfg["data"]
    if name == "physiocag2d":
        return PhysioCAG2D(in_channels=data.get("in_channels", 1),
                           num_classes=data.get("num_classes", 14), **m)
    if name == "physiocag3d":
        return PhysioCAG3D(in_channels=data.get("in_channels", 1),
                           num_classes=data.get("num_classes", 2), **m)
    raise ValueError(f"unknown model '{name}'")


def input_shape(cfg: Dict[str, Any], batch: int = 1) -> Tuple[int, ...]:
    data = cfg["data"]
    if cfg["model"]["name"] == "physiocag2d":
        s = data.get("image_size", 224)
        return (batch, data.get("in_channels", 1), s, s)
    return (batch, data.get("in_channels", 1), *data["spatial_size"])
