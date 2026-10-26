from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import torch
import yaml


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(pref: str = "auto") -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def set_by_path(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def _parse_value(text: str) -> Any:
    """YAML-parse an override value; YAML 1.1 reads '3e-3' as a string, so retry as float."""
    val = yaml.safe_load(text)
    if isinstance(val, str):
        try:
            return float(val)
        except ValueError:
            return val
    return val


def load_config(path: str, overrides: Iterable[str] = ()) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for item in overrides:
        key, sep, val = item.partition("=")
        if not sep:
            raise ValueError(f"override '{item}' must look like a.b.c=value")
        set_by_path(cfg, key.strip(), _parse_value(val))
    return cfg


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def to_py(obj: Any) -> Any:
    """Recursively convert numpy/torch scalars and arrays to plain Python."""
    if isinstance(obj, dict):
        return {str(k): to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_py(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def save_json(obj: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_py(obj), f, indent=2)
