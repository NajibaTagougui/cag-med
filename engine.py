from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import DISEASES, build_datasets
from .losses import build_loss
from .metrics import classification_metrics, segmentation_metrics_per_sample, summarise_segmentation
from .models import build_model
from .utils import count_parameters, get_device, save_json, set_seed, to_py


def _guides(batch: dict, enabled: bool, device: torch.device):
    if not enabled:
        return None
    return [batch["omics"].to(device), batch["clinical"].to(device)]


def _loader(ds, cfg: dict, shuffle: bool) -> DataLoader:
    d = cfg["data"]
    nw, bs = d.get("num_workers", 0), d.get("batch_size", 8)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=nw,
                      pin_memory=torch.cuda.is_available(), drop_last=shuffle and len(ds) > bs,
                      persistent_workers=nw > 0)


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, cfg: dict, device: torch.device,
             criterion: Optional[torch.nn.Module] = None) -> Dict[str, Any]:
    model.eval()
    task = cfg["task"]
    use_guides = cfg.get("guides", {}).get("enabled", False)
    amp = cfg["train"].get("amp", False) and device.type == "cuda"
    losses, logits_all, labels_all = [], [], []
    per_sample: Dict[str, list] = {}
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label" if task == "classification" else "mask"].to(device)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(x, _guides(batch, use_guides, device))
        if criterion is not None:
            losses.append(float(criterion(logits.float(), y)))
        if task == "classification":
            logits_all.append(logits.float().cpu())
            labels_all.append(y.cpu())
        else:
            for k, v in segmentation_metrics_per_sample(logits.float(), y).items():
                per_sample.setdefault(k, []).extend(v.cpu().tolist())
    if task == "classification":
        m = classification_metrics(torch.cat(logits_all), torch.cat(labels_all), DISEASES)
    else:
        m = summarise_segmentation(per_sample)
        m["per_sample_dice"] = per_sample["dice"]
    if losses:
        m["loss"] = float(np.mean(losses))
    return m


def _monitor_key(cfg: dict) -> str:
    return cfg["train"].get("monitor", "mean_auc" if cfg["task"] == "classification" else "dice")


def train(cfg: Dict[str, Any], resume: Optional[str] = None) -> Dict[str, Any]:
    set_seed(cfg.get("seed", 42), cfg["train"].get("deterministic", False))
    device = get_device(cfg["train"].get("device", "auto"))
    out_dir = Path(cfg["train"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(cfg, out_dir / "config.json")

    data = build_datasets(cfg)
    loaders = {"train": _loader(data["train"], cfg, True),
               "val": _loader(data["val"], cfg, False),
               "test": _loader(data["test"], cfg, False)}
    model = build_model(cfg).to(device)
    pos_weight = data.get("pos_weight")
    criterion = build_loss(cfg, pos_weight.to(device) if pos_weight is not None else None).to(device)
    n_params = count_parameters(model)
    tc = cfg["train"]
    use_guides = cfg.get("guides", {}).get("enabled", False)
    print(f"[{cfg.get('experiment', 'run')}] device={device} params={n_params:,} "
          f"train/val/test={len(data['train'])}/{len(data['val'])}/{len(data['test'])} "
          f"guides={'real' if use_guides else 'NULL (zeros)'}")

    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc.get("weight_decay", 1e-4))
    steps_per_epoch = max(1, len(loaders["train"]))
    total, warm = tc["epochs"] * steps_per_epoch, int(tc.get("warmup_epochs", 0) * steps_per_epoch)

    def lr_lambda(step: int) -> float:
        if step < warm:
            return (step + 1) / max(1, warm)
        prog = (step - warm) / max(1, total - warm)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    amp = tc.get("amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    key = _monitor_key(cfg)
    best, best_epoch, start, history = -float("inf"), 0, 1, []

    if resume:
        ck = torch.load(resume, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        best, best_epoch, start, history = ck["best"], ck["best_epoch"], ck["epoch"] + 1, ck["history"]
        print(f"resumed from {resume} at epoch {start}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    task = cfg["task"]
    for epoch in range(start, tc["epochs"] + 1):
        model.train()
        t0, run_loss, n_batches = time.time(), 0.0, 0
        for batch in loaders["train"]:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label" if task == "classification" else "mask"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(x, _guides(batch, use_guides, device))
            loss = criterion(logits.float(), y)
            scaler.scale(loss).backward()
            if tc.get("grad_clip", 1.0) and tc["grad_clip"] > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += float(loss.detach())
            n_batches += 1
        val = evaluate(model, loaders["val"], cfg, device, criterion)
        score = val.get(key, float("nan"))
        row = {"epoch": epoch, "train_loss": run_loss / max(1, n_batches),
               "val": {k: v for k, v in val.items() if k not in ("per_class_auc", "per_sample_dice")},
               "lr": opt.param_groups[0]["lr"], "seconds": time.time() - t0}
        history.append(row)
        flag = ""
        if not math.isnan(score) and score > best:
            best, best_epoch, flag = score, epoch, " *best*"
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch,
                        "metrics": to_py(val)}, out_dir / "best.pt")
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(), "epoch": epoch, "best": best,
                    "best_epoch": best_epoch, "history": to_py(history), "cfg": cfg},
                   out_dir / "last.pt")
        save_json(history, out_dir / "history.json")
        print(f"epoch {epoch:03d}/{tc['epochs']} loss={row['train_loss']:.4f} "
              f"val_{key}={score:.4f} ({row['seconds']:.0f}s){flag}")

    ck = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(ck["model"])
    test = evaluate(model, loaders["test"], cfg, device)
    results = {
        "experiment": cfg.get("experiment"), "task": task, "monitor": key,
        "best_val": best, "best_epoch": best_epoch, "test": test, "params": n_params,
        "peak_vram_gb": (torch.cuda.max_memory_allocated() / 1e9) if device.type == "cuda" else None,
        "guides": "real" if use_guides else "null (zeros): unimodal imaging baseline",
        "attention": cfg["model"].get("attention", "spatial"), "seed": cfg.get("seed", 42),
    }
    save_json(results, out_dir / "results.json")
    print(f"TEST {key}: {test.get(key, float('nan')):.4f}  -> {out_dir / 'results.json'}")
    return results
