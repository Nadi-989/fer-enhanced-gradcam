"""Model-parameter randomisation sanity check (Adebayo et al., 2018).

Weights are re-initialised cascading from the classifier head down to the
input. A trustworthy explanation method must change when the model it explains
is destroyed, so the Spearman rank correlation between the CAM of the trained
model and the CAM of the randomised model should fall towards 0.

    python sanity_check.py --run runs/fer2013_resnet18 --n_images 64
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from evaluate_cam import load_run, test_loader
from src.cams import compute_cams
from src.metrics import spearman
from src.utils import get_device, set_seed


def reinit(module: torch.nn.Module):
    for m in module.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.reset_running_stats()


def cascade_stages(model):
    """Top-level children with parameters, from output to input."""
    stages = [(n, c) for n, c in model.named_children() if any(True for _ in c.parameters())]
    return list(reversed(stages))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--methods", nargs="*")
    ap.add_argument("--n_images", type=int, default=64)
    args = ap.parse_args()
    run, device = Path(args.run), get_device()
    model, layers, cfg, classes = load_run(run, device)
    set_seed(cfg.get("seed", 42))
    methods = args.methods or cfg.get("cam_methods")
    loader, _ = test_loader(run, cfg, args.n_images, args.n_images, cfg.get("seed", 42))
    x, _, _ = next(iter(loader)); x = x.to(device)
    with torch.no_grad():
        tgt = model(x).argmax(1)

    ref = {m: compute_cams(m, model, layers, x, tgt) for m in methods}
    rows = []
    # Randomise a deep copy stage by stage; layer handles are looked up by name
    rand = copy.deepcopy(model)
    name_of = {id(mod): n for n, mod in model.named_modules()}
    rmods = dict(rand.named_modules())
    rlayers = {"deep": rmods[name_of[id(layers["deep"])]],
               "fusion": [rmods[name_of[id(l)]] for l in layers["fusion"]]}
    stages = cascade_stages(rand)
    randomised = []
    for name, stage in stages:
        reinit(stage); randomised.append(name)
        for m in methods:
            cams = compute_cams(m, rand, rlayers, x, tgt)
            rho = [spearman(ref[m][i], cams[i]) for i in range(x.size(0))]
            rows.append({"randomised_up_to": name, "n_stages": len(randomised), "method": m,
                         "spearman_mean": float(np.nanmean(rho)), "spearman_std": float(np.nanstd(rho))})
    df = pd.DataFrame(rows)
    df.to_csv(run / "sanity_check.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    for m, d in df.groupby("method", sort=False):
        ax.plot(d.randomised_up_to, d.spearman_mean, marker="o", label=m)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_ylabel("Spearman ρ (trained vs randomised)")
    ax.set_xlabel("Cascading randomisation (output → input)")
    ax.tick_params(axis="x", rotation=30); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(run / "sanity_check.png", dpi=200); plt.close(fig)
    order = [n for n, _ in stages]
    print(df.pivot(index="method", columns="randomised_up_to", values="spearman_mean")[order].round(3))


if __name__ == "__main__":
    main()
