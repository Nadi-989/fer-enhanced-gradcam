"""Helpers shared by train.py, test.py and draw.py: loading saved models and plotting."""
from __future__ import annotations

import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .models import build_model


def find_checkpoints(items: list[str]) -> list[Path]:
    """Accept best.pt files, wildcards or folders; return every best.pt found."""
    found = []
    for it in items:
        for p in sorted(glob.glob(it)) or [it]:
            p = Path(p)
            if p.is_dir():
                found += sorted(p.rglob("best.pt"))
            elif p.suffix == ".pt" and p.exists():
                found.append(p)
    if not found:
        raise SystemExit("No best.pt found. Pass a .pt file or a folder that contains one.")
    return list(dict.fromkeys(found))


def load_model(ckpt_path: str | Path, device) -> dict:
    """Rebuild the network named inside best.pt and load its trained weights (no training)."""
    ckpt_path = Path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg, classes = ckpt["cfg"], ckpt["classes"]
    model, layers = build_model(cfg["arch"], len(classes), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(True)  # gradient-based maps need this
    name = f"{cfg['arch']} ({Path(cfg.get('data_root', '')).name or ckpt_path.parent.name})"
    if ckpt_path.parent.name.startswith("fold"):
        name += f" {ckpt_path.parent.name}"
    return {"model": model, "layers": layers, "cfg": cfg, "classes": classes, "name": name,
            "run_dir": ckpt_path.parent, "best_epoch": ckpt.get("epoch")}


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
    for a in axes:
        a.set_xlabel("epoch")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
