"""Aggregate k-fold runs: accuracy/F1 mean +- std, pooled confusion matrix, pooled CAM metrics.

    python aggregate_cv.py --runs runs/ckplus48_resnet18/fold*
Writes cv_results.json, confusion_matrix_cv.png and cam_summary.* to the parent folder.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_cam import summarise
from train import plot_confusion
from src.utils import save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out")
    args = ap.parse_args()
    runs = sorted(Path(r) for r in args.runs if (Path(r) / "test_results.json").exists())
    if not runs:
        raise SystemExit("No fold folders with test_results.json found.")
    out = Path(args.out) if args.out else runs[0].parent
    classes = json.loads((runs[0] / "splits.json").read_text(encoding="utf-8"))["classes"]

    res = [json.loads((r / "test_results.json").read_text(encoding="utf-8")) for r in runs]
    acc = np.array([r["test_accuracy"] for r in res]); f1 = np.array([r["test_macro_f1"] for r in res])
    cm = np.sum([np.array(r["confusion_matrix"]) for r in res], axis=0)
    per_class_f1 = {c: float(np.mean([r["per_class"][c]["f1-score"] for r in res])) for c in classes}
    summary = {"folds": [str(r) for r in runs], "accuracy_mean": acc.mean(), "accuracy_std": acc.std(ddof=1) if len(acc) > 1 else 0.0,
               "macro_f1_mean": f1.mean(), "macro_f1_std": f1.std(ddof=1) if len(f1) > 1 else 0.0,
               "per_fold_accuracy": acc.tolist(), "per_class_f1_mean": per_class_f1,
               "pooled_confusion_matrix": cm.tolist()}
    save_json(summary, out / "cv_results.json")
    plot_confusion(cm, classes, out / "confusion_matrix_cv.png", f"{len(runs)}-fold CV (pooled)")
    print(f"Accuracy {acc.mean():.4f} ± {summary['accuracy_std']:.4f} | macro-F1 {f1.mean():.4f} ± {summary['macro_f1_std']:.4f}")

    cams = [r / "cam_metrics_per_image.csv" for r in runs if (r / "cam_metrics_per_image.csv").exists()]
    if cams:
        df = pd.concat([pd.read_csv(p).assign(fold=p.parent.name) for p in cams], ignore_index=True)
        df.to_csv(out / "cam_metrics_per_image.csv", index=False)
        summarise(df, out)
        print((out / "cam_summary.md").read_text(encoding="utf-8"))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
