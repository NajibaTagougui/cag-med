"""NIfTI volume datasets (BraTS 2020, MSD Heart) with on-disk caching of resized tensors."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .guides import NullGuides
from .transforms import augment_3d

Item = Dict[str, object]


def load_nifti(path) -> np.ndarray:
    import nibabel as nib

    return np.asarray(nib.load(str(path)).get_fdata(dtype=np.float32))


def to_dhw(arr: np.ndarray) -> np.ndarray:
    """NIfTI (x, y, z) -> (z, x, y)."""
    return np.ascontiguousarray(arr.transpose(2, 0, 1))


def zscore(vol: np.ndarray, nonzero_only: bool) -> np.ndarray:
    if not nonzero_only:
        return (vol - vol.mean()) / (vol.std() + 1e-8)
    mask = vol != 0
    out = np.zeros_like(vol)
    if mask.any():
        out[mask] = (vol[mask] - vol[mask].mean()) / (vol[mask].std() + 1e-8)
    return out


def resize(t: torch.Tensor, size: Sequence[int], mode: str) -> torch.Tensor:
    kw = {} if mode == "nearest" else {"align_corners": False}
    return F.interpolate(t.unsqueeze(0), size=tuple(size), mode=mode, **kw)[0]


def discover_brats(root: str) -> List[Item]:
    items: List[Item] = []
    for seg in sorted(Path(root).rglob("*_seg.nii*")):
        d = seg.parent
        mods = []
        for m in ("t1", "t1ce", "t2", "flair"):
            hits = sorted(d.glob(f"*_{m}.nii*"))
            if not hits:
                break
            mods.append(hits[0])
        if len(mods) == 4:
            items.append({"id": d.name, "images": mods, "label": seg})
    return items


def discover_msd_heart(root: str) -> List[Item]:
    root_p = Path(root)
    items: List[Item] = []
    for img in sorted((root_p / "imagesTr").glob("*.nii*")):
        if img.name.startswith("._"):
            continue
        lab = root_p / "labelsTr" / img.name
        if lab.exists():
            items.append({"id": img.name.split(".")[0], "images": [img], "label": lab})
    return items


def split_items(items: List[Item], n_train: Optional[int], n_val: int, n_test: int,
                seed: int) -> Dict[str, List[Item]]:
    """Deterministic subject-level split; the test/val sets do not depend on n_train."""
    order = np.random.RandomState(seed).permutation(len(items))
    shuffled = [items[i] for i in order]
    if n_val + n_test >= len(shuffled):
        raise ValueError(f"need more than {n_val + n_test} subjects, found {len(shuffled)}")
    test = shuffled[:n_test]
    val = shuffled[n_test:n_test + n_val]
    train = shuffled[n_test + n_val:]
    if n_train is not None:
        train = train[:n_train]
    return {"train": train, "val": val, "test": test}


class NiftiSegDataset(Dataset):
    def __init__(self, items: List[Item], spatial_size: Sequence[int], cache_dir: str,
                 nonzero_norm: bool = True, augment: bool = False, noise_std: float = 0.0,
                 guides=None):
        self.items = items
        self.size = tuple(spatial_size)
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.nonzero_norm = nonzero_norm
        self.augment, self.noise_std = augment, noise_std
        self.guides = guides or NullGuides()

    def __len__(self) -> int:
        return len(self.items)

    def _load(self, item: Item) -> Tuple[torch.Tensor, torch.Tensor]:
        tag = "x".join(map(str, self.size))
        path = self.cache / f"{item['id']}_{tag}_{int(self.nonzero_norm)}.pt"
        if path.exists():
            blob = torch.load(path)
            return blob["x"].float(), blob["y"].long()
        chans = [zscore(to_dhw(load_nifti(p)), self.nonzero_norm) for p in item["images"]]
        x = resize(torch.from_numpy(np.stack(chans)), self.size, "trilinear")
        y = (to_dhw(load_nifti(item["label"])) > 0).astype(np.float32)
        y = resize(torch.from_numpy(y).unsqueeze(0), self.size, "nearest")[0]
        torch.save({"x": x.half(), "y": y.byte()}, path)
        return x, y.long()

    def __getitem__(self, i: int) -> dict:
        item = self.items[i]
        x, y = self._load(item)
        if self.augment:
            x, y = augment_3d(x, y, noise_std=self.noise_std)
        omics, clinical = self.guides(str(item["id"]))
        return {"image": x, "mask": y, "omics": omics, "clinical": clinical,
                "id": str(item["id"])}
