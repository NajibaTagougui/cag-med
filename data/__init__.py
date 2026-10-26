from __future__ import annotations

from typing import Any, Dict

from .chestxray14 import DISEASES, ChestXray14, compute_pos_weights
from .guides import CSVGuides, NullGuides, build_guides
from .synthetic import synthetic_splits
from .volumes import NiftiSegDataset, discover_brats, discover_msd_heart, split_items

__all__ = ["DISEASES", "ChestXray14", "CSVGuides", "NullGuides", "NiftiSegDataset",
           "build_datasets", "compute_pos_weights", "discover_brats", "discover_msd_heart",
           "split_items", "build_guides"]


def build_datasets(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return {'train','val','test', optional 'pos_weight'} for the configured dataset."""
    d = cfg["data"]
    name = d["name"]
    seed = cfg.get("seed", 42)
    if name == "synthetic":
        return synthetic_splits(cfg)

    if name == "chestxray14":
        common = dict(root=d["root"], image_size=d.get("image_size", 224),
                      in_channels=d.get("in_channels", 1),
                      val_fraction=d.get("val_fraction", 0.1), seed=seed,
                      hflip=d.get("hflip", False), max_samples=d.get("max_samples"))
        train = ChestXray14(split="train", augment=d.get("augment", True),
                            guides=build_guides(cfg), **common)
        guides = build_guides(cfg, fit_ids=train.names)
        train.guides = guides
        out = {"train": train,
               "val": ChestXray14(split="val", guides=guides, **common),
               "test": ChestXray14(split="test", guides=guides, **common),
               "pos_weight": compute_pos_weights(train.labels, d.get("pos_weight_cap", 50.0))}
        return out

    if name in ("brats2020", "msd_heart"):
        items = discover_brats(d["root"]) if name == "brats2020" else discover_msd_heart(d["root"])
        if not items:
            raise RuntimeError(f"no subjects found under {d['root']}")
        parts = split_items(items, d.get("n_train"), d["n_val"], d["n_test"], seed)
        guides = build_guides(cfg, fit_ids=[str(i["id"]) for i in parts["train"]])
        mk = lambda key, aug: NiftiSegDataset(  # noqa: E731
            parts[key], d["spatial_size"], d.get("cache_dir", "cache"),
            nonzero_norm=d.get("nonzero_norm", name == "brats2020"),
            augment=aug, noise_std=d.get("noise_std", 0.0), guides=guides)
        return {"train": mk("train", d.get("augment", False)),
                "val": mk("val", False), "test": mk("test", False)}

    raise ValueError(f"unknown dataset '{name}'")
