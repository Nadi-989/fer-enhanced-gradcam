"""Shared helpers: config loading, seeding, normalisation, and heat-map overlays."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_config(path: str | os.PathLike, overrides: dict | None = None) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})
    cfg.setdefault("out_dir", f"runs/{Path(path).stem}_{cfg.get('arch', 'model')}")
    return cfg


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_json(obj, path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def denormalize(x: torch.Tensor) -> np.ndarray:
    """(3,H,W) normalised tensor -> (H,W,3) float RGB image in [0,1]."""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(3, 1, 1)
    img = (x * std + mean).clamp(0, 1)
    return img.permute(1, 2, 0).detach().cpu().numpy()


def overlay_cam(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a [0,1] CAM over an RGB [0,1] image.

    NOTE: cv2.applyColorMap returns **BGR**. Forgetting the BGR->RGB conversion
    swaps red and blue, so the most important region is shown in blue - a very
    common bug that makes a correct explanation look wrong.
    """
    heat = cv2.applyColorMap(np.uint8(255 * np.clip(cam, 0, 1)), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    out = (1 - alpha) * rgb + alpha * heat
    return np.clip(out, 0, 1)


def minmax(a: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    a = a - a.min()
    return a / (a.max() + eps)
