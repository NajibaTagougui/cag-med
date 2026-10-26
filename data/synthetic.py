"""Tiny synthetic datasets used ONLY for unit tests and CI smoke runs.

Results obtained on these datasets say nothing about real performance.
"""
from __future__ import annotations

import torch
from torch.utils.data import Dataset


class SyntheticChest(Dataset):
    def __init__(self, n: int = 64, image_size: int = 32, in_channels: int = 1,
                 num_classes: int = 14, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, in_channels, image_size, image_size, generator=g)
        self.y = (torch.rand(n, num_classes, generator=g) < 0.2).float()
        cell = image_size // 4
        for c in range(num_classes):                      # learnable signal per class
            r, col = (c // 4) % 4, c % 4
            idx = self.y[:, c] > 0
            self.x[idx, :, r * cell:(r + 1) * cell, col * cell:(col + 1) * cell] += 2.0
        self.ids = [f"syn_{i}" for i in range(n)]

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int) -> dict:
        return {"image": self.x[i], "label": self.y[i], "omics": torch.zeros(8),
                "clinical": torch.zeros(4), "id": self.ids[i]}


class SyntheticSeg3D(Dataset):
    def __init__(self, n: int = 8, spatial_size=(16, 24, 24), in_channels: int = 1, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        d, h, w = spatial_size
        self.x = torch.randn(n, in_channels, d, h, w, generator=g) * 0.5
        self.y = torch.zeros(n, d, h, w, dtype=torch.long)
        zz, yy, xx = torch.meshgrid(torch.arange(d), torch.arange(h), torch.arange(w),
                                    indexing="ij")
        for i in range(n):
            c = [int(torch.randint(4, s - 4, (1,), generator=g)) for s in (d, h, w)]
            ball = ((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2) <= 9
            self.y[i][ball] = 1
            self.x[i, 0][ball] += 2.0
        self.ids = [f"syn_{i}" for i in range(n)]

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int) -> dict:
        return {"image": self.x[i], "mask": self.y[i], "omics": torch.zeros(8),
                "clinical": torch.zeros(4), "id": self.ids[i]}


def synthetic_splits(cfg: dict) -> dict:
    d = cfg["data"]
    if cfg["task"] == "classification":
        mk = lambda n, s: SyntheticChest(n, d.get("image_size", 32), d.get("in_channels", 1),  # noqa: E731
                                         d.get("num_classes", 14), seed=s)
    else:
        mk = lambda n, s: SyntheticSeg3D(n, d["spatial_size"], d.get("in_channels", 1), seed=s)  # noqa: E731
    return {"train": mk(d.get("n_train", 32), 1), "val": mk(d.get("n_val", 8), 2),
            "test": mk(d.get("n_test", 8), 3)}
