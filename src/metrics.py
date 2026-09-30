"""Validation metrics for saliency maps.

Faithfulness (does the map point at what the *model* uses?)
  deletion_auc   remove most-salient pixels first; lower AUC = better   (Petsiuk et al., RISE)
  insertion_auc  add most-salient pixels to a blurred image; higher = better
  average_drop   % confidence lost when only the CAM-weighted image is kept; lower = better
  increase_conf  % images whose confidence rises under the CAM mask; higher = better
                 (Chattopadhay et al., Grad-CAM++)

Plausibility (does the map point where *humans/FACS* expect?)
  fres           Facial Region Energy Share: fraction of CAM energy inside the
                 AU regions of the explained emotion
  frcr           Facial Region Concentration Ratio = fres / area share of those
                 regions (1.0 = no better than uniform; >1 = concentrated)
  face_energy    fraction of CAM energy inside the face ellipse (vs background)
  pointing_hit   1 if the CAM maximum falls inside the emotion's AU regions
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur

from .regions import emotion_mask, face_mask

_trapz = getattr(np, "trapezoid", None) or np.trapz


# ------------------------------------------------------------ plausibility --
def region_scores(cam: np.ndarray, emotion: str) -> dict:
    h, w = cam.shape
    em, fm = emotion_mask(emotion, h, w), face_mask(h, w)
    total = cam.sum() + 1e-8
    fres = float(cam[em].sum() / total)
    area = em.mean()
    ys, xs = np.unravel_index(np.argmax(cam), cam.shape)
    return {
        "fres": fres,
        "frcr": fres / max(area, 1e-8),
        "face_energy": float(cam[fm].sum() / total),
        "pointing_hit": float(em[ys, xs]),
    }


# ------------------------------------------------------------ faithfulness --
@torch.no_grad()
def _probs(model, x, targets, bs=128):
    out = []
    for i in range(0, x.shape[0], bs):
        out.append(F.softmax(model(x[i:i + bs]), 1))
    p = torch.cat(out)
    return p.gather(1, targets.view(-1, 1)).squeeze(1)


@torch.no_grad()
def drop_increase(model, x, cams, targets) -> tuple[np.ndarray, np.ndarray]:
    """Per-image Average-Drop term and Increase-in-Confidence indicator."""
    cam_t = torch.from_numpy(cams).to(x.device).unsqueeze(1)
    p = _probs(model, x, targets)
    o = _probs(model, x * cam_t, targets)  # normalised space: masked pixels -> dataset mean
    drop = (torch.clamp(p - o, min=0) / (p + 1e-8)) * 100
    inc = (o > p).float() * 100
    return drop.cpu().numpy(), inc.cpu().numpy()


@torch.no_grad()
def deletion_insertion(model, x, cams, targets, steps: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Per-image deletion and insertion AUC (normalised to [0,1])."""
    b, c, h, w = x.shape
    blurred = gaussian_blur(x, kernel_size=[11, 11], sigma=[5.0, 5.0])
    zeros = torch.zeros_like(x)  # normalised-space zero == dataset mean colour
    n = h * w
    ks = np.linspace(0, n, steps + 1).astype(int)
    dels, ins = [], []
    for i in range(b):
        order = torch.from_numpy(np.argsort(-cams[i].reshape(-1), kind="stable").copy()).to(x.device)
        d_batch, i_batch = [], []
        for k in ks:
            keep = torch.zeros(n, device=x.device)
            keep[order[:k]] = 1.0
            keep = keep.view(1, h, w)
            d_batch.append(x[i] * (1 - keep) + zeros[i] * keep)
            i_batch.append(blurred[i] * (1 - keep) + x[i] * keep)
        t = targets[i].repeat(len(ks))
        pd = _probs(model, torch.stack(d_batch), t).cpu().numpy()
        pi = _probs(model, torch.stack(i_batch), t).cpu().numpy()
        dels.append(_trapz(pd, dx=1.0 / steps))
        ins.append(_trapz(pi, dx=1.0 / steps))
    return np.array(dels), np.array(ins)


# ------------------------------------------------------------- sanity check --
def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a.ravel())).astype(np.float64)
    rb = np.argsort(np.argsort(b.ravel())).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum()) + 1e-12
    return float((ra * rb).sum() / denom)
