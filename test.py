"""STEP 2 - TESTING ONLY. Loads trained best.pt files and measures them. No training.

A) Test a model on its own test images (the split saved by train.py):
    python test.py --models runs/fer2013_resnet18/best.pt
    python test.py --models runs                 # every best.pt under runs/

   Writes into each run folder: test_results.json (accuracy, macro-F1, per-class
   scores, confusion matrix) and confusion_matrix.png, and prints a summary table.

B) Classify new photos (no labels needed):
    python test.py --models runs/fer2013_resnet18/best.pt --images photos/*.jpg

   Writes predictions.csv (image, model, predicted emotion, confidence, top-3).
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

import cv2

from src.data import FaceDataset, build_transforms
from src.face import crop_face
from src.runtime import find_checkpoints, load_model, plot_confusion, predict
from src.utils import get_device, save_json


def test_split(m, device, batch_size):
    split_file = m["run_dir"] / "splits.json"
    if not split_file.exists():
        print(f"[skip] {m['name']}: no splits.json next to best.pt"); return None
    items = [tuple(t) for t in json.loads(split_file.read_text(encoding="utf-8"))["test"]]
    missing = [p for p, _ in items if not Path(p).exists()]
    if missing:
        print(f"[skip] {m['name']}: test images not found (e.g. {missing[0]}). Prepare the dataset first."); return None
    loader = DataLoader(FaceDataset(items, build_transforms(m["cfg"]["img_size"], False)),
                        batch_size=batch_size, shuffle=False, num_workers=m["cfg"].get("num_workers", 2))
    classes = m["classes"]
    y, p, _ = predict(m["model"], loader, device)
    cm = confusion_matrix(y, p, labels=list(range(len(classes))))
    res = {"arch": m["cfg"]["arch"], "best_epoch": m["best_epoch"], "n_test": len(items),
           "test_accuracy": accuracy_score(y, p), "test_macro_f1": f1_score(y, p, average="macro"),
           "test_weighted_f1": f1_score(y, p, average="weighted"),
           "per_class": classification_report(y, p, labels=list(range(len(classes))), target_names=classes,
                                              output_dict=True, zero_division=0),
           "confusion_matrix": cm.tolist()}
    save_json(res, m["run_dir"] / "test_results.json")
    plot_confusion(cm, classes, m["run_dir"] / "confusion_matrix.png", f"{m['name']} - test")
    return {"model": m["name"], "test images": len(items), "accuracy": round(res["test_accuracy"], 4),
            "macro-F1": round(res["test_macro_f1"], 4), "best epoch": m["best_epoch"]}


@torch.no_grad()
def classify_images(models, images, device, crop):
    rows = []
    for path in images:
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"[skip] cannot read {path}"); continue
        if crop:
            gray, found = crop_face(gray)
            if not found:
                print(f"[warn] no face detected in {path}; using the centre of the image")
        face = Image.fromarray(gray).convert("RGB")
        for m in models:
            x = build_transforms(m["cfg"]["img_size"], False)(face).unsqueeze(0).to(device)
            prob = torch.softmax(m["model"](x), 1)[0]
            top = prob.argsort(descending=True)[:3].tolist()
            rows.append({"image": path, "model": m["name"], "prediction": m["classes"][top[0]],
                         "confidence": round(float(prob[top[0]]), 4),
                         "top3": ", ".join(f"{m['classes'][i]} {prob[i]:.2f}" for i in top)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True, help="best.pt files or folders containing them")
    ap.add_argument("--images", nargs="*", help="new photos to classify (mode B)")
    ap.add_argument("--out", default="test_results", help="folder for predictions.csv / summary (mode B)")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--no_crop", action="store_true", help="images are already cropped faces")
    args = ap.parse_args()

    device = get_device()
    models = [load_model(p, device) for p in find_checkpoints(args.models)]
    print("Loaded models:", *[f"  - {m['name']}" for m in models], sep="\n")

    if args.images:
        images = [p for it in args.images for p in (sorted(glob.glob(it)) or [it])]
        df = classify_images(models, images, device, not args.no_crop)
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "predictions.csv", index=False)
        print(df.to_string(index=False)); print(f"Saved {out / 'predictions.csv'}")
        return

    rows = [r for m in models if (r := test_split(m, device, args.batch_size))]
    if rows:
        print(pd.DataFrame(rows).to_string(index=False))
        print("test_results.json and confusion_matrix.png written next to each best.pt")


if __name__ == "__main__":
    main()
