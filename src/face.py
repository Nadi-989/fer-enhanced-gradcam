"""Shared face detection / cropping (OpenCV Haar cascade, no downloads needed)."""
from __future__ import annotations

import cv2
import numpy as np

_CASCADE = None


def _cascade():
    global _CASCADE
    if _CASCADE is None:
        _CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return _CASCADE


def crop_face(gray: np.ndarray, margin: float = 0.08, out_size: int | None = None):
    """Return (square crop of the largest face, found?) from a grayscale uint8 image.

    Falls back to a centred square crop when no face is detected.
    """
    h, w = gray.shape[:2]
    faces = _cascade().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                        minSize=(max(24, min(h, w) // 8),) * 2)
    found = len(faces) > 0
    if found:
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        side = int(max(fw, fh) * (1 + 2 * margin))
        cx, cy = x + fw // 2, y + fh // 2
    else:
        side, cx, cy = min(h, w), w // 2, h // 2
    x0, y0 = max(0, cx - side // 2), max(0, cy - side // 2)
    x1, y1 = min(w, x0 + side), min(h, y0 + side)
    crop = gray[y0:y1, x0:x1]
    if out_size:
        crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return crop, found
