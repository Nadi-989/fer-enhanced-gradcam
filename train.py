"""Train an emotion classifier on the six basic emotions.

    python train.py --config configs/fer2013.yaml --arch resnet18
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score)
from tqdm import tqdm

from src.data import build_loaders, class_weights, describe, save_splits
from src.models import build_model
from src.utils import get_device, load_config, save_json, set_seed


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, ps, probs = [], [], []
    for x, y, _ in loader:
        out = torch.softmax(model(x.to(device)), 1)
        probs.append(out.cpu()); ps.append(out.argmax(1).cpu()); ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy(), torch.cat(probs).numpy()


def plot_confusion(cm, classes, path, title):
    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    fig.colorbar(im, fraction=0.046); fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def plot_history(hist: pd.DataFrame, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].plot(hist.epoch, hist.train_loss, label="train"); axes[0].plot(hist.epoch, hist.val_loss, label="val")
    axes[0].set_title("Loss"); axes[0].legend()
    axes[1].plot(hist.epoch, hist.train_acc, label="train"); axes[1].plot(hist.epoch, hist.val_acc, label="val")
    axes[1].plot(hist.epoch, hist.val_f1, label="val macro-F1"); axes[1].set_title("Accuracy / F1"); axes[1].legend()
    for a in axes: a.set_xlabel("epoch")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--arch"); ap.add_argument("--epochs", type=int); ap.add_argument("--out_dir")
    ap.add_argument("--batch_size", type=int); ap.add_argument("--lr", type=float)
    ap.add_argument("--pretrained", type=lambda s: s.lower() in ("1", "true", "yes"))
    ap.add_argument("--fold", type=int, help="fold index when the config sets n_folds")
    args = ap.parse_args()
    cfg = load_config(args.config, {k: v for k, v in vars(args).items() if k != "config"})
    if not args.out_dir:
        cfg["out_dir"] = f"runs/{Path(args.config).stem}_{cfg['arch']}"
        if cfg.get("n_folds"):
            cfg["out_dir"] += f"/fold{cfg.get('fold', 0)}"
    out = Path(cfg["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.get("seed", 42))
    device = get_device()

    loaders, splits = build_loaders(cfg)
    classes = cfg["classes"]
    save_splits(splits, classes, out)
    print("Split sizes:", describe(splits, classes))

    model, _ = build_model(cfg["arch"], len(classes), cfg.get("pretrained", True))
    model.to(device)
    weights = class_weights(splits["train"], len(classes)).to(device) if cfg.get("class_weights", True) else None
    crit = nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg.get("label_smoothing", 0.1))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-4))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_f1, bad, rows = -1.0, 0, []
    patience = cfg.get("patience", 10)
    for epoch in range(1, cfg["epochs"] + 1):
        model.train(); t0 = time.time()
        tl, tc, tn = 0.0, 0, 0
        for x, y, _ in tqdm(loaders["train"], desc=f"epoch {epoch}", leave=False):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x); loss = crit(logits, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tl += loss.item() * x.size(0); tc += (logits.argmax(1) == y).sum().item(); tn += x.size(0)
        sched.step()

        model.eval(); vl, vn = 0.0, 0
        with torch.no_grad():
            for x, y, _ in loaders["val"]:
                x, y = x.to(device), y.to(device)
                vl += nn.functional.cross_entropy(model(x), y).item() * x.size(0); vn += x.size(0)
        yv, pv, _ = predict(model, loaders["val"], device)
        row = dict(epoch=epoch, train_loss=tl / tn, train_acc=tc / tn, val_loss=vl / vn,
                   val_acc=accuracy_score(yv, pv), val_f1=f1_score(yv, pv, average="macro"),
                   lr=opt.param_groups[0]["lr"], sec=time.time() - t0)
        rows.append(row)
        print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
        if row["val_f1"] > best_f1:
            best_f1, bad = row["val_f1"], 0
            torch.save({"state_dict": model.state_dict(), "cfg": cfg, "classes": classes,
                        "epoch": epoch}, out / "best.pt")
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping at epoch {epoch}"); break

    hist = pd.DataFrame(rows); hist.to_csv(out / "history.csv", index=False)
    plot_history(hist, out / "training_curves.png")

    ckpt = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    yt, pt, _ = predict(model, loaders["test"], device)
    cm = confusion_matrix(yt, pt, labels=list(range(len(classes))))
    results = {
        "arch": cfg["arch"], "best_epoch": ckpt["epoch"],
        "test_accuracy": accuracy_score(yt, pt),
        "test_macro_f1": f1_score(yt, pt, average="macro"),
        "test_weighted_f1": f1_score(yt, pt, average="weighted"),
        "per_class": classification_report(yt, pt, labels=list(range(len(classes))),
                                           target_names=classes, output_dict=True, zero_division=0),
        "confusion_matrix": cm.tolist(),
    }
    save_json(results, out / "test_results.json")
    plot_confusion(cm, classes, out / "confusion_matrix.png", f"{cfg['arch']} - test")
    print(f"TEST acc={results['test_accuracy']:.4f} macroF1={results['test_macro_f1']:.4f} -> {out}")


if __name__ == "__main__":
    main()
