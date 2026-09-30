"""Dataset scanning, reproducible (optionally subject-independent) splits, loaders.

Expected layout (any dataset, class names are folder names):

    data_root/
        train/<class>/*.png     # optional: if train/ and test/ exist they are used
        val/<class>/*.png       # optional
        test/<class>/*.png
    or
    data_root/<class>/*.png     # single pool -> split with test_split / val_split

Folders whose (mapped) name is not in `classes` (e.g. `neutral`, `contempt`)
are ignored, which is how the six basic emotions are selected.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from .utils import IMAGENET_MEAN, IMAGENET_STD, save_json

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Common folder-name spellings mapped to the canonical six emotion names.
DEFAULT_ALIASES = {
    "anger": "angry", "angry": "angry",
    "disgust": "disgust", "disgusted": "disgust",
    "fear": "fear", "fearful": "fear",
    "happy": "happy", "happiness": "happy",
    "sad": "sad", "sadness": "sad",
    "surprise": "surprise", "surprised": "surprise",
}


def canonical(name: str, folder_map: dict | None = None) -> str:
    key = name.strip().lower()
    if folder_map and key in folder_map:
        return folder_map[key]
    return DEFAULT_ALIASES.get(key, key)


def scan_dir(root: Path, classes: list[str], folder_map: dict | None = None) -> list[tuple[str, int]]:
    items = []
    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        name = canonical(cls_dir.name, folder_map)
        if name not in classes:
            continue
        label = classes.index(name)
        for f in sorted(cls_dir.rglob("*")):
            if f.suffix.lower() in IMG_EXT:
                items.append((str(f), label))
    return items


def _groups(items, subject_regex: str | None):
    if not subject_regex:
        return None
    pat = re.compile(subject_regex)
    groups = []
    for path, _ in items:
        m = pat.search(Path(path).name)
        groups.append(m.group(1) if m else Path(path).name)
    return np.array(groups)


def _split(items, frac, seed, subject_regex=None):
    """Stratified split; subject-independent when subject_regex is given."""
    labels = np.array([l for _, l in items])
    groups = _groups(items, subject_regex)
    idx = np.arange(len(items))
    if groups is not None:
        n_splits = max(2, int(round(1 / frac)))
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        keep, hold = next(sgkf.split(idx, labels, groups))
    else:
        keep, hold = train_test_split(idx, test_size=frac, stratify=labels, random_state=seed)
    return [items[i] for i in keep], [items[i] for i in hold]


def _kfold(items, n_folds, fold, seed, subject_regex=None):
    """Stratified k-fold (subject-grouped when possible); returns (trainval, test) of `fold`."""
    labels = np.array([l for _, l in items])
    groups = _groups(items, subject_regex)
    idx = np.arange(len(items))
    if groups is not None:
        splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        folds = list(splitter.split(idx, labels, groups))
    else:
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        folds = list(splitter.split(idx, labels))
    keep, hold = folds[fold]
    return [items[i] for i in keep], [items[i] for i in hold]


def make_splits(cfg: dict) -> dict[str, list[tuple[str, int]]]:
    root = Path(cfg["data_root"])
    classes, fmap = cfg["classes"], cfg.get("folder_map")
    seed, rx = cfg.get("seed", 42), cfg.get("subject_regex")
    if not root.exists():
        raise FileNotFoundError(f"data_root not found: {root}. See README 'Datasets'.")

    if (root / "train").is_dir():
        trainval = scan_dir(root / "train", classes, fmap)
        test = scan_dir(root / "test", classes, fmap) if (root / "test").is_dir() else None
        if test is None:
            trainval, test = _split(trainval, cfg.get("test_split", 0.2), seed, rx)
        if (root / "val").is_dir():
            train, val = trainval, scan_dir(root / "val", classes, fmap)
        else:
            train, val = _split(trainval, cfg.get("val_split", 0.1), seed, rx)
    else:
        pool = scan_dir(root, classes, fmap)
        if cfg.get("n_folds"):
            trainval, test = _kfold(pool, cfg["n_folds"], cfg.get("fold", 0), seed, rx)
        else:
            trainval, test = _split(pool, cfg.get("test_split", 0.2), seed, rx)
        train, val = _split(trainval, cfg.get("val_split", 0.1), seed, rx)

    splits = {"train": train, "val": val, "test": test}
    for k, v in splits.items():
        if not v:
            raise RuntimeError(f"Empty '{k}' split - check data_root and class folder names.")
    return splits


def describe(splits, classes) -> dict:
    return {k: {classes[c]: n for c, n in sorted(Counter(l for _, l in v).items())}
            for k, v in splits.items()}


class FaceDataset(Dataset):
    """Loads faces as grayscale and replicates to 3 channels (keeps ImageNet weights usable)."""

    def __init__(self, items, transform):
        self.items, self.transform = items, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        img = Image.open(path).convert("L").convert("RGB")
        return self.transform(img), label, i


def build_transforms(img_size: int, train: bool):
    norm = [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if not train:
        return T.Compose([T.Resize((img_size, img_size)), *norm])
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(),
        T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        T.ColorJitter(brightness=0.25, contrast=0.25),
        *norm,
        T.RandomErasing(p=0.25, scale=(0.02, 0.08)),
    ])


def build_loaders(cfg: dict, splits=None):
    splits = splits or make_splits(cfg)
    size, bs, nw = cfg["img_size"], cfg["batch_size"], cfg.get("num_workers", 2)
    pin = torch.cuda.is_available()
    loaders = {
        "train": DataLoader(FaceDataset(splits["train"], build_transforms(size, True)),
                            batch_size=bs, shuffle=True, num_workers=nw, pin_memory=pin, drop_last=True),
        "val": DataLoader(FaceDataset(splits["val"], build_transforms(size, False)),
                          batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin),
        "test": DataLoader(FaceDataset(splits["test"], build_transforms(size, False)),
                           batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin),
    }
    return loaders, splits


def save_splits(splits, classes, out_dir):
    save_json({"classes": classes, **{k: v for k, v in splits.items()}},
              Path(out_dir) / "splits.json")


def class_weights(items, n_classes) -> torch.Tensor:
    counts = np.bincount([l for _, l in items], minlength=n_classes).astype(np.float64)
    w = counts.sum() / (n_classes * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32)
