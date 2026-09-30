"""Explain predictions on your own images (face is detected and cropped first).

    python predict.py --run runs/fer2013_resnet18 --images photo1.jpg photo2.png
Saves one figure per image to <run>/predictions/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image

from evaluate_cam import load_run
from src.cams import compute_cams
from src.data import build_transforms
from src.face import crop_face
from src.utils import denormalize, get_device, overlay_cam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--methods", nargs="*", default=["gradcam", "gradcam++", "layercam", "msf"])
    ap.add_argument("--no_crop", action="store_true", help="image is already a cropped face")
    args = ap.parse_args()
    run, device = Path(args.run), get_device()
    model, layers, cfg, classes = load_run(run, device)
    tf = build_transforms(cfg["img_size"], False)
    out_dir = run / "predictions"; out_dir.mkdir(exist_ok=True)

    for path in args.images:
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"[skip] cannot read {path}"); continue
        found = True
        if not args.no_crop:
            gray, found = crop_face(gray)
            if not found:
                print(f"[warn] no face detected in {path}; using centre crop")
        x = tf(Image.fromarray(gray).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = torch.softmax(model(x), 1)[0]
        pred = int(prob.argmax())
        tgt = torch.tensor([pred], device=device)
        rgb = denormalize(x[0])

        fig, axes = plt.subplots(1, len(args.methods) + 1, figsize=(2.4 * (len(args.methods) + 1), 2.8))
        axes[0].imshow(rgb); axes[0].set_title(f"{classes[pred]} ({prob[pred]:.2f})", fontsize=9)
        for ax, m in zip(axes[1:], args.methods):
            cam = compute_cams(m, model, layers, x, tgt)[0]
            ax.imshow(overlay_cam(rgb, cam)); ax.set_title(m, fontsize=9)
        for ax in axes: ax.axis("off")
        fig.tight_layout()
        dst = out_dir / f"{Path(path).stem}_explained.png"
        fig.savefig(dst, dpi=200); plt.close(fig)
        top = ", ".join(f"{classes[i]} {prob[i]:.2f}" for i in prob.argsort(descending=True)[:3].tolist())
        print(f"{path}: {top} -> {dst}")


if __name__ == "__main__":
    main()
