"""STEP 3 - DRAWING ONLY. Loads trained best.pt files and draws the heat maps. No training.

Examples
    # one model
    python draw.py --models runs/fer2013_resnet18/best.pt --images face.jpg

    # several networks at once (compared side by side, one row per model)
    python draw.py --models runs/fer2013_resnet18/best.pt runs/ckplus48_vgg16/fold0/best.pt \
                   runs/ckplus48_efficientnet_b0/fold0/best.pt --images photos/*.jpg

    # a folder: every best.pt under it is used
    python draw.py --models runs --images face.jpg

Outputs (in --out, default: drawings/)
    <image>_compare.png   rows = models, columns = input + one map per method
    predictions.csv       image, model, predicted emotion, confidence, top-3
"""
from __future__ import annotations

import argparse
import glob
import os

os.environ.setdefault("TQDM_DISABLE", "1")  # hide Score-CAM progress bars
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image

from src.cams import compute_cams
from src.data import build_transforms
from src.face import crop_face
from src.runtime import find_checkpoints, load_model
from src.utils import denormalize, get_device, overlay_cam


def load_face(path: str, crop: bool):
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None, False
    found = True
    if crop:
        gray, found = crop_face(gray)
    return Image.fromarray(gray).convert("RGB"), found


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True, help="best.pt files or folders containing them")
    ap.add_argument("--images", nargs="+", required=True, help="image files (wildcards allowed)")
    ap.add_argument("--methods", nargs="*", default=["gradcam", "scorecam", "msf"])
    ap.add_argument("--out", default="drawings")
    ap.add_argument("--no_crop", action="store_true", help="images are already cropped faces")
    args = ap.parse_args()

    device = get_device()
    models = [load_model(p, device) for p in find_checkpoints(args.models)]
    print("Loaded models:", *[f"  - {m['name']}" for m in models], sep="\n")
    images = [p for it in args.images for p in (sorted(glob.glob(it)) or [it])]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    rows = []
    for img_path in images:
        face, found = load_face(img_path, not args.no_crop)
        if face is None:
            print(f"[skip] cannot read {img_path}"); continue
        if not found:
            print(f"[warn] no face detected in {img_path}; using the centre of the image")

        k = len(args.methods) + 1
        fig, axes = plt.subplots(len(models), k, figsize=(2.3 * k, 2.5 * len(models)), squeeze=False)
        for r, m in enumerate(models):
            x = build_transforms(m["cfg"]["img_size"], False)(face).unsqueeze(0).to(device)
            with torch.no_grad():
                prob = torch.softmax(m["model"](x), 1)[0]
            pred = int(prob.argmax())
            top = prob.argsort(descending=True)[:3].tolist()
            rows.append({"image": img_path, "model": m["name"], "prediction": m["classes"][pred],
                         "confidence": round(float(prob[pred]), 4),
                         "top3": ", ".join(f"{m['classes'][i]} {prob[i]:.2f}" for i in top)})
            rgb = denormalize(x[0])
            axes[r, 0].imshow(rgb)
            axes[r, 0].set_title(f"{m['classes'][pred]} ({prob[pred]:.2f})", fontsize=9)
            axes[r, 0].set_ylabel(m["name"], fontsize=8)
            tgt = torch.tensor([pred], device=device)
            for c, method in enumerate(args.methods, 1):
                cam = compute_cams(method, m["model"], m["layers"], x, tgt)[0]
                axes[r, c].imshow(overlay_cam(rgb, cam))
                if r == 0:
                    axes[r, c].set_title(method, fontsize=9)
        for a in axes.ravel():
            a.set_xticks([]); a.set_yticks([])
        fig.tight_layout()
        dst = out / f"{Path(img_path).stem}_compare.png"
        fig.savefig(dst, dpi=200); plt.close(fig)
        print(f"{img_path} -> {dst}")

    df = pd.DataFrame(rows)
    df.to_csv(out / "predictions.csv", index=False)
    if not df.empty:
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
