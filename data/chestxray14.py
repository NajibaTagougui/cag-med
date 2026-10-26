"""NIH ChestX-ray14 (112,120 frontal radiographs, 14 labels).

Expected layout (Kaggle / NIH): ``root`` contains ``Data_Entry_2017.csv``,
``train_val_list.txt``, ``test_list.txt`` and the PNGs anywhere below it
(e.g. ``images_001/images/*.png``). The official test list is used as-is. The
train/val split is done **by patient** inside train_val_list to avoid leakage between
images of the same patient.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .guides import NullGuides
from .transforms import augment_2d

DISEASES = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
            "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis",
            "Pleural_Thickening", "Hernia"]


@lru_cache(maxsize=4)
def _index_images(root: str) -> Dict[str, Path]:
    return {p.name: p for p in Path(root).rglob("*.png")}


def compute_pos_weights(labels: np.ndarray, cap: float = 50.0) -> torch.Tensor:
    pos = labels.sum(0)
    neg = len(labels) - pos
    w = np.where(pos > 0, neg / np.maximum(pos, 1), 1.0)
    return torch.tensor(np.clip(w, 1.0, cap), dtype=torch.float32)


class ChestXray14(Dataset):
    def __init__(self, root: str, split: str, image_size: int = 224, in_channels: int = 1,
                 val_fraction: float = 0.1, seed: int = 42, augment: bool = False,
                 hflip: bool = False, mean: float = 0.5, std: float = 0.25,
                 guides=None, max_samples: Optional[int] = None):
        import pandas as pd
        from PIL import Image  # noqa: F401  (fail early if Pillow is missing)

        if split not in ("train", "val", "test"):
            raise ValueError("split must be train|val|test")
        root_p = Path(root)
        df = pd.read_csv(root_p / "Data_Entry_2017.csv")
        read = lambda n: set((root_p / n).read_text().split())  # noqa: E731
        if split == "test":
            df = df[df["Image Index"].isin(read("test_list.txt"))]
        else:
            df = df[df["Image Index"].isin(read("train_val_list.txt"))]
            patients = np.array(sorted(df["Patient ID"].unique()))
            rng = np.random.RandomState(seed)
            rng.shuffle(patients)
            n_val = int(round(len(patients) * val_fraction))
            val_patients = set(patients[:n_val].tolist())
            in_val = df["Patient ID"].isin(val_patients)
            df = df[in_val] if split == "val" else df[~in_val]

        index = _index_images(str(root_p))
        df = df[df["Image Index"].isin(index.keys())].reset_index(drop=True)
        if max_samples:
            df = df.iloc[:max_samples]
        if len(df) == 0:
            raise RuntimeError(f"no images found for split '{split}' under {root}")

        tags = df["Finding Labels"].fillna("No Finding").str.split("|")
        self.labels = np.stack(
            [[float(d in t) for d in DISEASES] for t in tags]).astype(np.float32)
        self.names: List[str] = df["Image Index"].tolist()
        self.paths = [index[n] for n in self.names]
        self.image_size, self.in_channels = image_size, in_channels
        self.mean, self.std = mean, std
        self.augment, self.hflip = augment and split == "train", hflip
        self.guides = guides or NullGuides()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> dict:
        from PIL import Image

        img = Image.open(self.paths[i]).convert("L").resize(
            (self.image_size, self.image_size), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).unsqueeze(0)
        x = (x - self.mean) / self.std
        if self.augment:
            x = augment_2d(x, hflip=self.hflip)
        if self.in_channels > 1:
            x = x.repeat(self.in_channels, 1, 1)
        omics, clinical = self.guides(self.names[i])
        return {"image": x, "label": torch.from_numpy(self.labels[i]),
                "omics": omics, "clinical": clinical, "id": self.names[i]}
