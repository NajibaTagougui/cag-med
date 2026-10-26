from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from .utils import count_parameters, get_device, load_config, save_json


def _cfg(args):
    return load_config(args.config, args.set)


def cmd_train(args):
    from .engine import train

    train(_cfg(args), resume=args.resume)


def cmd_evaluate(args):
    from .data import build_datasets
    from .engine import _loader, evaluate
    from .models import build_model

    cfg = _cfg(args)
    device = get_device(cfg["train"].get("device", "auto"))
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)["model"])
    ds = build_datasets(cfg)[args.split]
    res = evaluate(model, _loader(ds, cfg, False), cfg, device)
    out = Path(args.ckpt).with_name(f"eval_{args.split}.json")
    save_json(res, out)
    print({k: v for k, v in res.items() if k not in ("per_class_auc", "per_sample_dice")})
    print("saved", out)


def cmd_gatemaps(args):
    from .data import build_datasets
    from .models import build_model
    from .visualize import collect_gates, plot_gate_overlay

    cfg = _cfg(args)
    device = get_device(cfg["train"].get("device", "auto"))
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)["model"])
    ds = build_datasets(cfg)[args.split]
    real = cfg.get("guides", {}).get("enabled", False)
    if not real:
        print("WARNING: guides are null; every gate equals 0.5 by construction.")
    out = Path(args.out)
    for i in range(min(args.num, len(ds))):
        s = ds[i]
        x = s["image"].unsqueeze(0).to(device)
        guides = [s["omics"].unsqueeze(0).to(device), s["clinical"].unsqueeze(0).to(device)] if real else None
        _, maps = collect_gates(model, x, guides)
        mask = s["mask"].numpy() if "mask" in s else None
        plot_gate_overlay(s["image"].numpy(), [m[0].numpy() for m in maps], str(out / f"{s['id']}.png"),
                          mask=mask, guides_real=real)
        gates = {f"gate{k + 1}": m[0].numpy() for k, m in enumerate(maps)}
        np.savez_compressed(out / f"{s['id']}.npz", **gates)
    print("saved to", out)


def cmd_benchmark(args):
    from torch.utils.flop_counter import FlopCounterMode

    from .models import build_model, input_shape

    cfg = _cfg(args)
    device = get_device(args.device)
    model = build_model(cfg).to(device).eval()
    x = torch.randn(*input_shape(cfg, args.batch_size), device=device)
    with torch.no_grad():
        for _ in range(3):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for _ in range(args.runs):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        ms = (time.time() - t0) / args.runs * 1e3
        try:
            with FlopCounterMode(display=False) as fc:
                model(x)
            gflops = fc.get_total_flops() / 1e9
        except Exception:  # pragma: no cover
            gflops = float("nan")
    res = {"params": count_parameters(model), "latency_ms": ms, "gflops": gflops,
           "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else None,
           "device": str(device), "batch_size": args.batch_size}
    print(res)


def cmd_params(args):
    from .models import build_model

    m = build_model(_cfg(args))
    print(f"trainable parameters: {count_parameters(m):,}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="physiocag")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--config", required=True)
        sp.add_argument("--set", nargs="*", default=[], help="overrides, e.g. train.epochs=5")

    t = sub.add_parser("train"); common(t); t.add_argument("--resume"); t.set_defaults(fn=cmd_train)  # noqa: E702
    e = sub.add_parser("evaluate"); common(e); e.add_argument("--ckpt", required=True)  # noqa: E702
    e.add_argument("--split", default="test", choices=["train", "val", "test"])
    e.set_defaults(fn=cmd_evaluate)
    g = sub.add_parser("gatemaps"); common(g); g.add_argument("--ckpt", required=True)  # noqa: E702
    g.add_argument("--out", default="gatemaps"); g.add_argument("--num", type=int, default=4)  # noqa: E702
    g.add_argument("--split", default="test"); g.set_defaults(fn=cmd_gatemaps)  # noqa: E702
    b = sub.add_parser("benchmark"); common(b); b.add_argument("--batch-size", type=int, default=1)  # noqa: E702
    b.add_argument("--runs", type=int, default=20); b.add_argument("--device", default="auto")  # noqa: E702
    b.set_defaults(fn=cmd_benchmark)
    c = sub.add_parser("params"); common(c); c.set_defaults(fn=cmd_params)  # noqa: E702

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
