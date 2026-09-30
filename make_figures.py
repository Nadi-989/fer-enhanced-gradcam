"""Paper figures.

    python make_figures.py --run runs/fer2013_resnet18 --qualitative --per_class 2
    python make_figures.py --run runs/fer2013_resnet18 --regions
    python make_figures.py --run runs/fer2013_resnet18 --bars       # needs evaluate_cam.py output
Figures go to <run>/figures/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from evaluate_cam import load_run
from src.cams import compute_cams
from src.data import build_transforms
from src.regions import EMOTION_REGIONS, emotion_mask
from src.utils import denormalize, get_device, overlay_cam


def pick_images(model, items, tf, device, per_class, n_classes, seed):
    """Correctly classified, most confident test images per class (fallback: any)."""
    rng = np.random.default_rng(seed)
    chosen = []
    for c in range(n_classes):
        cand = [it for it in items if it[1] == c]
        cand = [cand[i] for i in rng.permutation(len(cand))[:200]]
        if not cand:
            continue
        x = torch.stack([tf(Image.open(p).convert("L").convert("RGB")) for p, _ in cand]).to(device)
        with torch.no_grad():
            prob = torch.softmax(model(x), 1)
        conf = prob[:, c].cpu().numpy(); ok = (prob.argmax(1) == c).cpu().numpy()
        order = np.lexsort((-conf, ~ok))
        chosen += [cand[i] for i in order[:per_class]]
    return chosen


def qualitative(run, model, layers, cfg, classes, items, methods, per_class, device, figs):
    tf = build_transforms(cfg["img_size"], False)
    chosen = pick_images(model, items, tf, device, per_class, len(classes), cfg.get("seed", 42))
    x = torch.stack([tf(Image.open(p).convert("L").convert("RGB")) for p, _ in chosen]).to(device)
    with torch.no_grad():
        pred = model(x).argmax(1)
    cams = {m: compute_cams(m, model, layers, x, pred) for m in methods}
    n, k = len(chosen), len(methods) + 1
    fig, axes = plt.subplots(n, k, figsize=(1.6 * k, 1.65 * n))
    axes = np.atleast_2d(axes)
    for i, (_, lab) in enumerate(chosen):
        rgb = denormalize(x[i])
        axes[i, 0].imshow(rgb)
        axes[i, 0].set_ylabel(f"{classes[lab]}\n→{classes[int(pred[i])]}", fontsize=7)
        for j, m in enumerate(methods, 1):
            axes[i, j].imshow(overlay_cam(rgb, cams[m][i]))
            if i == 0:
                axes[i, j].set_title(m, fontsize=8)
        if i == 0:
            axes[i, 0].set_title("input", fontsize=8)
    for a in axes.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(pad=0.3)
    fig.savefig(figs / "qualitative_comparison.png", dpi=250); plt.close(fig)


def regions(cfg, classes, items, figs):
    size = cfg["img_size"]
    tf = build_transforms(size, False)
    sample = [tf(Image.open(p).convert("L").convert("RGB")) for p, _ in items[:300]]
    mean_face = denormalize(torch.stack(sample).mean(0))
    emos = [c for c in classes if c in EMOTION_REGIONS]
    fig, axes = plt.subplots(1, len(emos), figsize=(1.9 * len(emos), 2.2))
    for ax, e in zip(np.atleast_1d(axes), emos):
        m = emotion_mask(e, size, size).astype(float)
        ax.imshow(mean_face); ax.imshow(np.ma.masked_where(m == 0, m), cmap="autumn", alpha=0.45)
        ax.set_title(f"{e}\n{'+'.join(EMOTION_REGIONS[e])}", fontsize=7); ax.axis("off")
    fig.tight_layout(); fig.savefig(figs / "emotion_regions.png", dpi=250); plt.close(fig)


def bars(run, figs):
    s = pd.read_csv(run / "cam_summary.csv")
    s = s[s.subset == "all"]
    metrics = [("deletion_auc", "Deletion AUC ↓"), ("insertion_auc", "Insertion AUC ↑"),
               ("average_drop", "Average Drop % ↓"), ("increase_conf", "Increase in Conf. % ↑"),
               ("frcr", "FRCR ↑"), ("pointing_hit", "Pointing game ↑")]
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6))
    for ax, (c, title) in zip(axes.ravel(), metrics):
        colors = ["#c0392b" if m.startswith("msf") else "#7f8c8d" for m in s.method]
        ax.bar(s.method, s[c], yerr=s[c + "_std"], color=colors, capsize=2)
        ax.set_title(title, fontsize=9); ax.tick_params(axis="x", rotation=45, labelsize=7)
    fig.tight_layout(); fig.savefig(figs / "cam_metrics_bars.png", dpi=250); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--qualitative", action="store_true")
    ap.add_argument("--regions", action="store_true")
    ap.add_argument("--bars", action="store_true")
    ap.add_argument("--methods", nargs="*")
    ap.add_argument("--per_class", type=int, default=2)
    args = ap.parse_args()
    run, device = Path(args.run), get_device()
    figs = run / "figures"; figs.mkdir(exist_ok=True)
    model, layers, cfg, classes = load_run(run, device)
    items = [tuple(t) for t in json.loads((run / "splits.json").read_text(encoding="utf-8"))["test"]]
    methods = args.methods or ["gradcam", "gradcam++", "scorecam", "layercam", "msf"]
    if not (args.qualitative or args.regions or args.bars):
        args.qualitative = args.regions = True
    if args.qualitative:
        qualitative(run, model, layers, cfg, classes, items, methods, args.per_class, device, figs)
    if args.regions:
        regions(cfg, classes, items, figs)
    if args.bars:
        bars(run, figs)
    print(f"Figures saved to {figs}")


if __name__ == "__main__":
    main()
