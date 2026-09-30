"""Facial regions and the emotion -> region prior used for validation.

Faces are assumed to be cropped and roughly aligned (true for FER2013 and for
CK+/RAF-DB after `scripts/prepare_ckplus.py` / aligned releases). Regions are
elliptical templates in normalised coordinates; check them visually with
`python make_figures.py --regions` before trusting the metric on a new dataset.

Emotion -> region mapping follows the prototypical Action Units of the six
basic emotions in FACS/EMFACS (Ekman & Friesen):
  happy     AU6 + AU12               -> cheeks, mouth
  sad       AU1 + AU4 + AU15         -> brows, mouth
  surprise  AU1 + AU2 + AU5 + AU26   -> brows, eyes, mouth
  fear      AU1+2+4+5+7+20+26        -> brows, eyes, mouth
  angry     AU4 + AU5 + AU7 + AU23   -> brows, eyes, mouth
  disgust   AU9 + AU15 + AU16        -> nose, mouth
"""
from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

# (x0, y0, x1, y1) boxes in [0,1]; each is drawn as the inscribed ellipse.
REGION_BOXES = {
    "brows": [(0.12, 0.16, 0.88, 0.34)],
    "eyes": [(0.12, 0.30, 0.88, 0.48)],
    "nose": [(0.36, 0.40, 0.64, 0.70)],
    "cheeks": [(0.10, 0.46, 0.36, 0.72), (0.64, 0.46, 0.90, 0.72)],
    "mouth": [(0.26, 0.64, 0.74, 0.88)],
}

EMOTION_REGIONS = {
    "happy": ["cheeks", "mouth"],
    "sad": ["brows", "mouth"],
    "surprise": ["brows", "eyes", "mouth"],
    "fear": ["brows", "eyes", "mouth"],
    "angry": ["brows", "eyes", "mouth"],
    "disgust": ["nose", "mouth"],
}

FACE_ELLIPSE = (0.5, 0.53, 0.46, 0.52)  # cx, cy, ax, ay (normalised)


def _draw(mask, box, h, w):
    x0, y0, x1, y1 = box
    center = (int(round((x0 + x1) / 2 * w)), int(round((y0 + y1) / 2 * h)))
    axes = (max(1, int(round((x1 - x0) / 2 * w))), max(1, int(round((y1 - y0) / 2 * h))))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1, -1)


@lru_cache(maxsize=256)
def region_mask(region: str, h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    for box in REGION_BOXES[region]:
        _draw(m, box, h, w)
    return m.astype(bool)


@lru_cache(maxsize=64)
def emotion_mask(emotion: str, h: int, w: int) -> np.ndarray:
    if emotion not in EMOTION_REGIONS:
        raise KeyError(f"No region prior for '{emotion}'. Known: {list(EMOTION_REGIONS)}")
    m = np.zeros((h, w), bool)
    for r in EMOTION_REGIONS[emotion]:
        m |= region_mask(r, h, w)
    return m


@lru_cache(maxsize=16)
def face_mask(h: int, w: int) -> np.ndarray:
    cx, cy, ax, ay = FACE_ELLIPSE
    m = np.zeros((h, w), np.uint8)
    cv2.ellipse(m, (int(cx * w), int(cy * h)), (int(ax * w), int(ay * h)), 0, 0, 360, 1, -1)
    return m.astype(bool)
