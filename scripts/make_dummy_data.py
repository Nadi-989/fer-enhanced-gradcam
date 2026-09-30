"""Tiny synthetic 'faces' for a smoke test of the whole pipeline (no real data).

Each class moves a different facial part (mouth curve, brow height, eye size),
so a model can learn it and CAMs should land on that part.

    python scripts/make_dummy_data.py && python train.py --config configs/dummy.yaml
"""
from pathlib import Path

import cv2
import numpy as np

CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise"]


def face(cls, rng, s=96):
    img = np.full((s, s), rng.integers(20, 60), np.uint8)
    cv2.ellipse(img, (s // 2, s // 2 + 3), (40, 46), 0, 0, 360, int(rng.integers(140, 180)), -1)
    j = lambda: int(rng.integers(-2, 3))
    brow_y = {"angry": 30, "surprise": 22, "fear": 24, "sad": 27}.get(cls, 27) + j()
    tilt = {"angry": 10, "sad": -10}.get(cls, 0)
    for cx, sgn in ((32, 1), (64, -1)):
        cv2.line(img, (cx - 10, brow_y - sgn * tilt // 3), (cx + 10, brow_y + sgn * tilt // 3), 40, 3)
        r = {"surprise": 7, "fear": 6}.get(cls, 4)
        cv2.circle(img, (cx + j(), 38 + j()), r, 30, -1)
    if cls == "disgust":
        cv2.line(img, (42, 52), (54, 52), 60, 2)
    my = 72 + j()
    if cls == "happy":
        cv2.ellipse(img, (48, my - 4), (16, 9), 0, 0, 180, 30, 3)
    elif cls == "sad":
        cv2.ellipse(img, (48, my + 5), (14, 7), 0, 180, 360, 30, 3)
    elif cls in ("surprise", "fear"):
        cv2.ellipse(img, (48, my), (8, 10 if cls == "surprise" else 6), 0, 0, 360, 30, -1)
    else:
        cv2.line(img, (38, my), (58, my), 30, 3)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    noise = rng.normal(0, 8, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def main(out="data/dummy", n_train=60, n_test=15, seed=0):
    rng = np.random.default_rng(seed)
    for split, n in (("train", n_train), ("test", n_test)):
        for c in CLASSES:
            d = Path(out) / split / c; d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                cv2.imwrite(str(d / f"{c}_{i:03d}.png"), face(c, rng))
    print(f"Synthetic data written to {out}")


if __name__ == "__main__":
    main()
