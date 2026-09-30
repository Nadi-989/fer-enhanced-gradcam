"""Build a six-emotion, face-cropped CK+ dataset.

Input: the official CK+ release
    <ck_root>/cohn-kanade-images/S005/001/S005_001_00000011.png
    <ck_root>/Emotion/S005/001/S005_001_00000011_emotion.txt   (label 0-7)
Output:
    <out>/<emotion>/S005_001_00000011.png   (grayscale, face-cropped, square)

The last `--frames` frames (peak expression) of each labelled sequence are
kept. Contempt (label 2) is dropped to keep the six basic emotions. File names
start with the subject id, so configs/ckplus.yaml can make subject-independent
splits (no person appears in both train and test).

    python scripts/prepare_ckplus.py --ck_root /path/to/CK+ --out data/ckplus
"""
import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.face import crop_face  # noqa: E402

CK_LABELS = {1: "angry", 3: "disgust", 4: "fear", 5: "happy", 6: "sad", 7: "surprise"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ck_root", required=True)
    ap.add_argument("--out", default="data/ckplus")
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--size", type=int, default=224)
    a = ap.parse_args()
    root, out = Path(a.ck_root), Path(a.out)
    imgs_root, emo_root = root / "cohn-kanade-images", root / "Emotion"
    written, missed = 0, 0
    for lab_file in sorted(emo_root.rglob("*_emotion.txt")):
        label = int(float(lab_file.read_text().strip()))
        if label not in CK_LABELS:
            continue
        seq_dir = imgs_root / lab_file.parent.relative_to(emo_root)
        frames = sorted(seq_dir.glob("*.png"))[-a.frames:]
        dst = out / CK_LABELS[label]; dst.mkdir(parents=True, exist_ok=True)
        for f in frames:
            gray = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            crop, found = crop_face(gray, out_size=a.size)
            missed += int(not found)
            cv2.imwrite(str(dst / f.name), crop); written += 1
    print(f"Wrote {written} images to {out} ({missed} without a detected face, centre-cropped)")


if __name__ == "__main__":
    main()
