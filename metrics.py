from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score


def classification_metrics(logits: torch.Tensor, labels: torch.Tensor,
                           class_names: List[str] | None = None) -> Dict:
    probs = torch.sigmoid(logits.float()).numpy()
    y = labels.numpy()
    aucs = []
    for c in range(y.shape[1]):
        if 0 < y[:, c].sum() < len(y):
            aucs.append(float(roc_auc_score(y[:, c], probs[:, c])))
        else:
            aucs.append(float("nan"))
    pred = (probs > 0.5).astype(np.float32)
    out = {
        "mean_auc": float(np.nanmean(aucs)) if not np.all(np.isnan(aucs)) else float("nan"),
        "n_valid_classes": int(np.sum(~np.isnan(aucs))),
        "accuracy": float((pred == y).mean()),            # element-wise over all labels
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
    }
    names = class_names or [str(i) for i in range(y.shape[1])]
    out["per_class_auc"] = dict(zip(names, aucs))
    return out


def segmentation_metrics_per_sample(logits: torch.Tensor, target: torch.Tensor,
                                    eps: float = 1e-8) -> Dict[str, torch.Tensor]:
    """Per-sample Dice/precision/recall/specificity for binary segmentation."""
    pred = (logits.argmax(1) > 0).flatten(1).float()
    tgt = (target > 0).flatten(1).float()
    tp = (pred * tgt).sum(1)
    fp = (pred * (1 - tgt)).sum(1)
    fn = ((1 - pred) * tgt).sum(1)
    tn = ((1 - pred) * (1 - tgt)).sum(1)
    return {
        "dice": (2 * tp + eps) / (2 * tp + fp + fn + eps),
        "precision": (tp + eps) / (tp + fp + eps),
        "recall": (tp + eps) / (tp + fn + eps),
        "specificity": (tn + eps) / (tn + fp + eps),
    }


def summarise_segmentation(per_sample: Dict[str, List[float]]) -> Dict[str, float]:
    out = {}
    for k, v in per_sample.items():
        arr = np.asarray(v, dtype=np.float64)
        out[k] = float(arr.mean())
        out[f"{k}_std"] = float(arr.std())
    out["n_samples"] = int(len(per_sample["dice"]))
    return out
