"""Run every CAM method on the test set and compute the validation metrics.

    python evaluate_cam.py --run runs/fer2013_resnet18
Outputs (in the run folder):
    cam_metrics_per_image.csv   one row per (image, method)
    cam_summary.csv / .md       mean +- std per method (all / correctly classified)
    cam_frcr_per_emotion.csv    plausibility per method x emotion
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.cams import compute_cams
from src.data import FaceDataset, build_transforms
from src.metrics import deletion_insertion, drop_increase, region_scores
from src.models import build_model
from src.utils import get_device, load_config, set_seed

import json


def load_run(run_dir: Path, device):
    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    cfg, classes = ckpt["cfg"], ckpt["classes"]
    model, layers = build_model(cfg["arch"], len(classes), pretrained=False)
    model.load_state_dict(ckpt["state_dict"]); model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(True)
    return model, layers, cfg, classes


def test_loader(run_dir: Path, cfg, bs, max_images, seed):
    splits = json.loads((run_dir / "splits.json").read_text(encoding="utf-8"))
    items = [tuple(t) for t in splits["test"]]
    ds = FaceDataset(items, build_transforms(cfg["img_size"], False))
    if max_images and max_images < len(ds):
        # class-stratified subsample for faster evaluation
        rng = np.random.default_rng(seed)
        labels = np.array([l for _, l in items])
        per = max(1, max_images // len(set(labels)))
        idx = np.concatenate([rng.permutation(np.where(labels == c)[0])[:per] for c in np.unique(labels)])
        ds = Subset(ds, np.sort(idx).tolist())
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=cfg.get("num_workers", 2)), items


def summarise(df: pd.DataFrame, out: Path):
    cols = ["deletion_auc", "insertion_auc", "average_drop", "increase_conf",
            "fres", "frcr", "face_energy", "pointing_hit", "ms_per_image"]
    rows = []
    for subset, d in (("all", df), ("correct", df[df.correct == 1])):
        g = d.groupby("method")[cols]
        m, s = g.mean(), g.std()
        for method in m.index:
            r = {"subset": subset, "method": method, "n": int(g.size()[method])}
            for c in cols:
                r[c] = m.loc[method, c]; r[c + "_std"] = s.loc[method, c]
            rows.append(r)
    summ = pd.DataFrame(rows)
    summ.to_csv(out / "cam_summary.csv", index=False)

    arrows = {"deletion_auc": "↓", "insertion_auc": "↑", "average_drop": "↓", "increase_conf": "↑",
              "fres": "↑", "frcr": "↑", "face_energy": "↑", "pointing_hit": "↑", "ms_per_image": "↓"}
    lines = []
    for subset in ("all", "correct"):
        d = summ[summ.subset == subset]
        lines += [f"### Test images: {subset}", "",
                  "| Method | " + " | ".join(f"{c} {arrows[c]}" for c in cols) + " |",
                  "|---|" + "---|" * len(cols)]
        for _, r in d.iterrows():
            lines.append(f"| {r.method} | " + " | ".join(
                f"{r[c]:.3f} ± {r[c + '_std']:.3f}" if c != "ms_per_image" else f"{r[c]:.1f}" for c in cols) + " |")
        lines.append("")
    (out / "cam_summary.md").write_text("\n".join(lines), encoding="utf-8")

    pe = df.pivot_table(index="method", columns="target_name", values="frcr", aggfunc="mean")
    pe.to_csv(out / "cam_frcr_per_emotion.csv")
    significance(df, out)
    return summ


def significance(df: pd.DataFrame, out: Path, ref: str = "gradcam"):
    """Two-sided Wilcoxon signed-rank test of every method against `ref` on per-image scores."""
    from scipy.stats import wilcoxon
    if ref not in set(df.method):
        return
    cols = ["deletion_auc", "insertion_auc", "average_drop", "fres", "frcr", "face_energy"]
    key = ["fold", "index"] if "fold" in df.columns else ["index"]
    base = df[df.method == ref].set_index(key)
    rows = []
    for m in sorted(set(df.method) - {ref}):
        d = df[df.method == m].set_index(key).reindex(base.index)
        for c in cols:
            a, b = d[c].to_numpy(), base[c].to_numpy()
            ok = ~(np.isnan(a) | np.isnan(b))
            try:
                p = wilcoxon(a[ok], b[ok]).pvalue if ok.sum() > 5 and np.any(a[ok] != b[ok]) else np.nan
            except ValueError:
                p = np.nan
            rows.append({"method": m, "vs": ref, "metric": c, "n": int(ok.sum()),
                         "median_diff": float(np.median(a[ok] - b[ok])), "p_value": p})
    pd.DataFrame(rows).to_csv(out / "significance_vs_gradcam.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run folder containing best.pt and splits.json")
    ap.add_argument("--methods", nargs="*")
    ap.add_argument("--max_images", type=int, help="stratified subsample of the test set")
    ap.add_argument("--target", choices=["pred", "true"], default="pred",
                    help="explain the predicted class (default) or the ground-truth class")
    ap.add_argument("--steps", type=int, default=20, help="deletion/insertion steps")
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    run = Path(args.run); device = get_device()
    model, layers, cfg, classes = load_run(run, device)
    set_seed(cfg.get("seed", 42))
    methods = args.methods or cfg.get("cam_methods")
    max_images = args.max_images if args.max_images is not None else cfg.get("cam_max_images")
    loader, items = test_loader(run, cfg, args.batch_size, max_images, cfg.get("seed", 42))

    records = []
    for x, y, idx in tqdm(loader, desc="CAM evaluation"):
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            pred = model(x).argmax(1)
        tgt = pred if args.target == "pred" else y
        for method in methods:
            t0 = time.time()
            cams = compute_cams(method, model, layers, x, tgt)
            if device.type == "cuda":
                torch.cuda.synchronize()
            ms = (time.time() - t0) * 1000 / x.size(0)
            drop, inc = drop_increase(model, x, cams, tgt)
            dele, ins = deletion_insertion(model, x, cams, tgt, steps=args.steps)
            for b in range(x.size(0)):
                name = classes[int(tgt[b])]
                records.append({
                    "index": int(idx[b]), "path": items[int(idx[b])][0], "method": method,
                    "true": classes[int(y[b])], "pred": classes[int(pred[b])], "target_name": name,
                    "correct": int(pred[b] == y[b]), "deletion_auc": dele[b], "insertion_auc": ins[b],
                    "average_drop": drop[b], "increase_conf": inc[b], "ms_per_image": ms,
                    **region_scores(cams[b], name),
                })
    df = pd.DataFrame(records)
    df.to_csv(run / "cam_metrics_per_image.csv", index=False)
    summ = summarise(df, run)
    print((run / "cam_summary.md").read_text(encoding="utf-8"))
    print(f"Saved results to {run}")


if __name__ == "__main__":
    main()
