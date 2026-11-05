"""Robustness evaluation: mean corruption accuracy (mCA).

Applies the Hendrycks corruption suite (conwaynet.corruptions) on the fly to the clean test
set at several severities, and reports per-corruption accuracy, the full mCA (all corruptions),
and the noise-mCA (gaussian/shot/impulse — the headline metric for the MNIST family). Uses the
same generators the CIFAR-10-C images were built from, so no precomputed corruption set is
needed.
"""
import numpy as np
import torch

from . import corruptions as C
from .data import DATASET_CONFIGS, _BG_COLOR, _FG_COLOR

NOISE = ("gaussian_noise", "shot_noise", "impulse_noise")


def _raw_eval_images(data_path, data_name, n_samples):
    """Return (images_u8 [(H,W,3) uint8], labels, channels) for the clean test split.

    Color variants are colorized here (blue bg / gold ink) so corruptions act on the image the
    model actually sees; grayscale stays 1-channel but is expanded to RGB for the corruption op.
    """
    cfg = DATASET_CONFIGS[data_name]
    ds = cfg["cls"](root=data_path, train=False, download=True, transform=None, **cfg["kwargs"])
    idx = np.linspace(0, len(ds) - 1, min(n_samples, len(ds))).astype(int)
    imgs, labels = [], []
    for i in idx:
        img, y = ds[int(i)]
        a = np.array(img.convert("L") if cfg["channels"] == 1 or cfg["color"] else img.convert("RGB"))
        if cfg["color"]:                              # colorize grayscale -> RGB uint8
            g = a.astype(np.float32) / 255.0
            bg, fg = np.array(_BG_COLOR), np.array(_FG_COLOR)
            a = ((bg + g[..., None] * (fg - bg)) * 255).clip(0, 255).astype(np.uint8)
        elif a.ndim == 2:                             # grayscale -> 3ch for the corruption op
            a = np.repeat(a[..., None], 3, axis=2)
        imgs.append(a)
        labels.append(int(y))
    return imgs, np.array(labels), cfg["channels"]


def _to_model_tensor(a_u8, channels, mean, std):
    """(H,W,3) uint8 -> normalized (C,H,W) tensor the model expects."""
    t = torch.from_numpy(a_u8).float().div(255.0).permute(2, 0, 1)   # (3,H,W)
    if channels == 1:
        t = t[:1]                                                     # grayscale: one channel's corruption (as MNIST-C), not the mean of 3 independent draws
    m = torch.tensor(mean).view(-1, 1, 1)
    s = torch.tensor(std).view(-1, 1, 1)
    return (t - m) / s


@torch.no_grad()
def _accuracy(model, tensors, labels, device, batch_size):
    model.eval()
    correct = 0
    for i in range(0, len(tensors), batch_size):
        x = torch.stack(tensors[i:i + batch_size]).to(device)
        y = torch.tensor(labels[i:i + batch_size]).to(device)
        correct += (model(x).argmax(1) == y).sum().item()
    return correct / len(tensors)


@torch.no_grad()
def evaluate_mca(model, data_path, data_name, mean, std, device, *,
                 corruptions=C.CORRUPTIONS_15, n_severities=5, n_samples=2000,
                 batch_size=256, seed=0):
    """Return {corruption: [acc per severity], 'clean': acc, 'mCA': float, 'noise_mCA': float}."""
    imgs, labels, channels = _raw_eval_images(data_path, data_name, n_samples)
    rng = np.random.RandomState(seed)

    clean = _accuracy(model, [_to_model_tensor(a, channels, mean, std) for a in imgs],
                      labels, device, batch_size)
    out = {"clean": clean}
    for name in corruptions:
        levels = min(n_severities, C.n_levels(name, extended=False))
        accs = []
        for lvl in range(levels):
            tens = [_to_model_tensor(C.apply(a, name, lvl, np.random.RandomState(rng.randint(1 << 30))),
                                     channels, mean, std) for a in imgs]
            accs.append(_accuracy(model, tens, labels, device, batch_size))
        out[name] = accs

    all_accs = [a for name in corruptions for a in out[name]]
    noise_accs = [a for name in NOISE if name in out for a in out[name]]
    out["mCA"] = float(np.mean(all_accs))
    out["noise_mCA"] = float(np.mean(noise_accs))
    return out
