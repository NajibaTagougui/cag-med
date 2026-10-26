"""Guide-vector providers (omics, clinical vitals).

The default provider returns zero vectors (a null guide), which makes any run a unimodal
imaging baseline. ``CSVGuides`` reads real per-sample vectors from a CSV so the CAG gates
are actually exercised.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


class NullGuides:
    def __init__(self, omics_dim: int = 8, clinical_dim: int = 4):
        self.omics_dim, self.clinical_dim = omics_dim, clinical_dim
        self.is_null = True

    def __call__(self, sample_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(self.omics_dim), torch.zeros(self.clinical_dim)


class CSVGuides:
    """Looks up guide vectors by sample id.

    Columns are z-scored with statistics from ``fit_ids`` (pass the training ids to avoid
    leakage; if omitted, all rows are used). Missing ids map to zero vectors and are counted
    in ``n_missing``.
    """

    def __init__(self, path: str, id_col: str, omics_cols: Sequence[str],
                 clinical_cols: Sequence[str], fit_ids: Optional[Sequence[str]] = None):
        import pandas as pd

        df = pd.read_csv(path)
        cols = list(omics_cols) + list(clinical_cols)
        missing_cols = [c for c in cols + [id_col] if c not in df.columns]
        if missing_cols:
            raise KeyError(f"columns not found in {path}: {missing_cols}")
        df[id_col] = df[id_col].astype(str)
        fit = df[df[id_col].isin(set(map(str, fit_ids)))] if fit_ids is not None else df
        mean = fit[cols].mean()
        std = fit[cols].std().replace(0, 1.0).fillna(1.0)
        z = ((df[cols] - mean) / std).fillna(0.0).astype(np.float32)
        n_o = len(omics_cols)
        self.table: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
            sid: (row[:n_o], row[n_o:]) for sid, row in zip(df[id_col], z.to_numpy())
        }
        self.omics_dim, self.clinical_dim = n_o, len(clinical_cols)
        self.n_missing = 0
        self.is_null = False

    def __call__(self, sample_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
        hit = self.table.get(str(sample_id))
        if hit is None:
            self.n_missing += 1
            return torch.zeros(self.omics_dim), torch.zeros(self.clinical_dim)
        return torch.from_numpy(hit[0].copy()), torch.from_numpy(hit[1].copy())


def build_guides(cfg: dict, fit_ids: Optional[List[str]] = None):
    g = cfg.get("guides", {}) or {}
    if not g.get("enabled", False):
        return NullGuides(g.get("omics_dim", 8), g.get("clinical_dim", 4))
    return CSVGuides(g["csv"], g["id_col"], g["omics_cols"], g["clinical_cols"], fit_ids)
