"""STEP 1 - TRAINING ONLY. Trains a network once and saves it as best.pt.

    python train.py --config configs/fer2013.yaml --arch resnet18

Saved in the run folder (runs/<config>_<arch>/):
    best.pt              trained weights + network name + settings (used by test.py and draw.py)
    splits.json          which images were train / val / test
    history.csv          loss and accuracy per epoch
    training_curves.png  the same, as a plot
Next:  python test.py --models runs/<run>/best.pt
       python draw.py --models runs/<run>/best.pt --images face.jpg
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from src.data import build_loaders, class_weights, describe, save_splits
from src.models import build_model
from src.runtime import plot_history, predict
from src.utils import get_device, load_config, set_seed


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

    print(f"Training finished. Best model (epoch {max(rows, key=lambda r: r['val_f1'])['epoch']}) saved to {out / 'best.pt'}")
    print(f"Next: python test.py --models {out / 'best.pt'}")

if __name__ == "__main__":
    main()
