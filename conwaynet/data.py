"""Datasets: MNIST family (B&W), their colorized 3-channel variants and CIFAR-10.

Three-way split. The torchvision TEST split is a genuine held-out set, touched only by the
final mCA evaluation. The validation set (used for monitoring during training) is carved
STRATIFIED out of the torchvision TRAIN split with a fixed `split_seed`, from a *second*
dataset instance that carries the eval transform — so validation/test never inherit the train
augmentation (crop/flip/AugMix). The split is recorded in `data_split_indices.json` with the
val indices and a hash of the train indices (CLAUDE.md §9); a later call with the same `output_path`
checks the recorded split and raises if it differs, so a changed split is detected, not overwritten.

Inputs: for the B&W and colorized families mean = std = 0.5, so (x-0.5)/0.5 lies in [-1, 1] and the
occupancy (x+1)/2 that the gate of the first layer reads is the raw intensity (black = 0). CIFAR-10
configs pass the standard per-channel statistics, which give roughly [-2, 2.1]; that is harmless
because in the WRN no gate reads the image (a stem convolution comes first).
"""
import hashlib
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# default (mean, std) normalisation per channel count
_NORM_GRAY = ((0.5,), (0.5,))
_NORM_RGB = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

# blue background / gold ink — the colorization used for the *_COLOR variants
_BG_COLOR = (0.10, 0.20, 0.85)
_FG_COLOR = (0.95, 0.60, 0.10)

# name -> dict(cls: torchvision class, kwargs: split kwargs, num_classes, channels, norm: default (mean, std),
#              color: colorize the 1-channel image blue/gold). All images are square (meta["image_size"]).
DATASET_CONFIGS = {
    "MNIST":              dict(cls=datasets.MNIST,        kwargs={},                     num_classes=10, channels=1, norm=_NORM_GRAY, color=False),
    "FashionMNIST":       dict(cls=datasets.FashionMNIST, kwargs={},                     num_classes=10, channels=1, norm=_NORM_GRAY, color=False),
    "KMNIST":             dict(cls=datasets.KMNIST,       kwargs={},                     num_classes=10, channels=1, norm=_NORM_GRAY, color=False),
    "EMNIST":             dict(cls=datasets.EMNIST,       kwargs={"split": "balanced"},  num_classes=47, channels=1, norm=_NORM_GRAY, color=False),
    "MNIST_COLOR":        dict(cls=datasets.MNIST,        kwargs={},                     num_classes=10, channels=3, norm=_NORM_RGB,  color=True),
    "KMNIST_COLOR":       dict(cls=datasets.KMNIST,       kwargs={},                     num_classes=10, channels=3, norm=_NORM_RGB,  color=True),
    "FashionMNIST_COLOR": dict(cls=datasets.FashionMNIST, kwargs={},                     num_classes=10, channels=3, norm=_NORM_RGB,  color=True),
    "EMNIST_COLOR":       dict(cls=datasets.EMNIST,       kwargs={"split": "balanced"},  num_classes=47, channels=3, norm=_NORM_RGB,  color=True),
    "CIFAR10":            dict(cls=datasets.CIFAR10,      kwargs={},                     num_classes=10, channels=3, norm=_NORM_RGB,  color=False),
}


class Colorize:
    """Map a 1-channel [0,1] image to a 3-channel image by linear interpolation between a
    background and a foreground color: out = bg + x * (fg - bg)."""

    def __init__(self, bg_color, fg_color):
        self.bg = torch.tensor(bg_color).view(-1, 1, 1)
        self.fg = torch.tensor(fg_color).view(-1, 1, 1)

    def __call__(self, x):
        return self.bg + x * (self.fg - self.bg)


def _build_transform(cfg, mean, std, train, normalize):
    ops = []
    # CIFAR-10 gets the standard crop+flip augmentation; the MNIST family gets none.
    if cfg["cls"] is datasets.CIFAR10 and train:
        ops += [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    ops.append(transforms.ToTensor())
    if cfg["color"]:
        ops.append(Colorize(_BG_COLOR, _FG_COLOR))
    if normalize:
        ops.append(transforms.Normalize(mean, std))     # AugMix, when on, normalizes after mixing
    return transforms.Compose(ops)


def _stratified_val_carve(labels, val_frac, split_seed):
    """Split indices [0..len) into (train_idx, val_idx), holding out `val_frac` per class."""
    labels = np.asarray(labels)
    rng = np.random.RandomState(split_seed)
    train_idx, val_idx = [], []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        n_val = int(round(len(idx) * val_frac))
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    train_idx.sort(); val_idx.sort()
    return train_idx, val_idx


def _write_split_sentinel(output_path, data_name, train_idx, val_idx, test_len, val_frac, split_seed):
    """Record the split in <output_path>/data_split_indices.json. If the file exists (a rerun, a resume, or an
    analysis reading a run folder), check that the split is the same instead of overwriting it."""
    if output_path is None:
        return
    os.makedirs(output_path, exist_ok=True)
    train_hash = hashlib.sha1(np.asarray(train_idx, dtype=np.int64).tobytes()).hexdigest()
    path = os.path.join(output_path, "data_split_indices.json")
    if os.path.exists(path):
        with open(path) as f:
            prev = json.load(f)
        if prev["dataset"] != data_name or prev["train_indices_sha1"] != train_hash or prev["val_indices"] != val_idx:
            raise RuntimeError(f"data split differs from the one recorded in {path} (dataset {prev['dataset']}, "
                               f"val_frac {prev['val_frac']}, split_seed {prev['split_seed']}); refusing to overwrite it")
        return
    with open(path, "w") as f:
        json.dump({"split_type": "stratified_val_carve_from_train__canonical_test_heldout",
                   "dataset": data_name, "val_frac": val_frac, "split_seed": split_seed,
                   "train_len": len(train_idx), "val_len": len(val_idx), "test_len": test_len,
                   "train_indices_sha1": train_hash, "val_indices": val_idx}, f, indent=2)


def load_dataset(data_path, batch_size, data_name, *, val_frac, split_seed, data_norms=None,
                 output_path=None, num_workers=4, shuffle_train=True, shuffle_val=False,
                 augmix=False, generator=None):
    """Return (train_loader, val_loader, test_loader, meta). meta has channels/num_classes/image_size/mean/std.

    Three-way split: `test_loader` is the torchvision TEST split (held out, eval transform);
    `val_loader` is a stratified `val_frac` carve of the torchvision TRAIN split, taken from a
    second dataset instance with the eval transform so it never sees the train augmentation;
    `train_loader` is the remaining train, with the train transform. `data_norms` = (mean, std)
    overrides the registry default. When `augmix` is True the TRAIN images are left un-normalized
    in [0,1] (AugMix runs in the training loop and normalizes afterwards); val/test are normalized.
    """
    if data_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset {data_name!r}; choose from {list(DATASET_CONFIGS)}")
    cfg = DATASET_CONFIGS[data_name]
    if not 0.0 < val_frac < 1.0:
        raise ValueError(f"val_frac must be in (0, 1), got {val_frac}")
    mean, std = data_norms if data_norms is not None else cfg["norm"]
    if len(mean) != cfg["channels"] or len(std) != cfg["channels"]:
        raise ValueError(f"data_norms has {len(mean)}/{len(std)} entries; {data_name} has {cfg['channels']} channel(s)")

    train_tf = _build_transform(cfg, mean, std, train=True, normalize=not augmix)
    eval_tf = _build_transform(cfg, mean, std, train=False, normalize=True)
    train_aug = cfg["cls"](root=data_path, train=True, download=True, transform=train_tf, **cfg["kwargs"])
    train_eval = cfg["cls"](root=data_path, train=True, download=True, transform=eval_tf, **cfg["kwargs"])
    test_set = cfg["cls"](root=data_path, train=False, download=True, transform=eval_tf, **cfg["kwargs"])
    labels = train_aug.targets

    train_idx, val_idx = _stratified_val_carve(labels, val_frac, split_seed)
    train_set = Subset(train_aug, train_idx)        # train transform (augmented)
    val_set = Subset(train_eval, val_idx)           # eval transform (no aug leak)

    _write_split_sentinel(output_path, data_name, train_idx, val_idx, len(test_set), val_frac, split_seed)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle_train,
                              num_workers=num_workers, pin_memory=True, generator=generator)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=shuffle_val,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    sample, _ = test_set[0]
    meta = dict(channels=cfg["channels"], num_classes=cfg["num_classes"],
                image_size=sample.shape[-1], mean=tuple(mean), std=tuple(std))
    return train_loader, val_loader, test_loader, meta
