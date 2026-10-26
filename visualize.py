"""Extract and plot REAL gate maps sigma(Omega) from a trained model.

Note: with null guides (zeros) every gate is exactly 0.5 by construction, so maps are only
informative when the model was trained and evaluated with real guide vectors.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .models.cag import CrossAttentionGate


@torch.no_grad()
def collect_gates(model: torch.nn.Module, x: torch.Tensor,
                  guides: Optional[Sequence[torch.Tensor]] = None
                  ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Run the model and return (output, [gate map per CAG stage, shape [B, *spatial]])."""
    gates = [m for m in model.modules() if isinstance(m, CrossAttentionGate)]
    if not gates:
        raise RuntimeError("model has no CAG modules")
    for g in gates:
        g.store_gate = True
    model.eval()
    out = model(x, guides)
    maps = []
    for g in gates:
        gm = g.last_gate.mean(1, keepdim=True)
        mode = "bilinear" if x.dim() == 4 else "trilinear"
        gm = F.interpolate(gm, size=x.shape[2:], mode=mode, align_corners=False)
        maps.append(gm[:, 0].cpu())
        g.store_gate = False
        g.last_gate = None
    return out, maps


def plot_gate_overlay(image: np.ndarray, gate_maps: Sequence[np.ndarray], out_path: str,
                      mask: Optional[np.ndarray] = None, titles: Optional[Sequence[str]] = None,
                      guides_real: bool = False, channel: int = 0) -> None:
    """image: [C,H,W] or [C,D,H,W]; gate_maps: list of [H,W] or [D,H,W] arrays."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if image.ndim == 4:                                   # pick the most informative axial slice
        d = int(np.argmax(mask.sum((1, 2)))) if mask is not None and mask.any() else image.shape[1] // 2
        img = image[channel, d]
        gate_maps = [g[d] for g in gate_maps]
        msk = mask[d] if mask is not None else None
    else:
        img, msk = image[channel], (mask if mask is not None else None)
    titles = titles or [f"CAG{i + 1} gate" for i in range(len(gate_maps))]
    fig, axes = plt.subplots(1, len(gate_maps) + 1, figsize=(3.4 * (len(gate_maps) + 1), 3.6))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("input")
    if msk is not None and msk.any():
        axes[0].contour(msk, levels=[0.5], colors="r", linewidths=1)
    for ax, g, t in zip(axes[1:], gate_maps, titles):
        ax.imshow(img, cmap="gray")
        im = ax.imshow(g, cmap="magma", vmin=0, vmax=1, alpha=0.6)
        ax.set_title(f"{t} (mean {g.mean():.3f})")
    for ax in axes:
        ax.axis("off")
    fig.colorbar(im, ax=axes.tolist(), fraction=0.02, label="gate value")
    fig.suptitle("guides: real" if guides_real else "guides: NULL (zeros) -> gate == 0.5 by construction",
                 fontsize=9)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
