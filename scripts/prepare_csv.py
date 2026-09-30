"""Convert a FER-style CSV (emotion, pixels, Usage) to class folders.

Works for the original fer2013.csv and for the Kaggle CK+48 'ckextended.csv'
(same format, plus label 7 = contempt). Only needed for CSV releases. The Kaggle *folder* release (train/, test/ with
one sub-folder per emotion) can be used directly: `neutral` is ignored by the
config, which keeps the six basic emotions.

    python scripts/prepare_csv.py --csv fer2013.csv     --out data/fer2013
    python scripts/prepare_csv.py --csv ckextended.csv  --out data/ckplus48 --pool
Usage column -> split:  Training -> train, PublicTest -> val, PrivateTest -> test
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

FER_LABELS = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "sad", 5: "surprise", 6: "neutral",
              7: "contempt"}
SPLIT = {"Training": "train", "PublicTest": "val", "PrivateTest": "test"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="data/fer2013")
    ap.add_argument("--keep_neutral", action="store_true")
    ap.add_argument("--pool", action="store_true",
                    help="ignore Usage and write one pool (<out>/<class>/) to be split by the config; "
                         "recommended for small sets such as CK+")
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    out = Path(a.out); n = 0
    for i, row in df.iterrows():
        name = FER_LABELS[int(row.emotion)]
        if name == "contempt" or (name == "neutral" and not a.keep_neutral):
            continue
        d = out / name if a.pool else out / SPLIT[row.Usage] / name
        d.mkdir(parents=True, exist_ok=True)
        px = np.array(row.pixels.split(), dtype=np.uint8)
        side = int(round(np.sqrt(px.size)))
        img = px.reshape(side, side)
        Image.fromarray(img).save(d / f"{i:05d}.png"); n += 1
    print(f"Wrote {n} images to {out}")


if __name__ == "__main__":
    main()
