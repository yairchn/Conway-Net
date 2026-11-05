"""Noise-robustness metric reported in the paper, as a pure function of a model + a normalized test batch
(so run.py and every analysis share the same code):

- `noise_mca_curve`: accuracy at [clean, severity 1..9], averaged over gaussian/shot/impulse using
  the 9-level severity ladders in `corruptions.py`.
- `noise_resilience_curve`: accuracy vs noise FRACTION t in [0,1], where the input is the convex blend
  x_t = (1-t)*x + t*uniform_noise (appendix breakdowns). At t=1 the image is pure uniform noise, so every model
  falls to chance.
"""
import zlib

import numpy as np
import torch

from . import corruptions as C

NOISE = ("gaussian_noise", "shot_noise", "impulse_noise")


@torch.no_grad()
def accuracy(model, X, Y, bs=512):
    model.eval()
    correct = 0
    for i in range(0, len(X), bs):
        out = model(X[i:i + bs])
        out = out[0] if isinstance(out, tuple) else out
        correct += (out.argmax(1) == Y[i:i + bs]).sum().item()
    return correct / len(X)


@torch.no_grad()
def noise_resilience_curve(model, X, Y, mean, std, n_points=11, seed=0):
    """Accuracy vs noise fraction t: x_t = (1-t)*x01 + t*U[0,1] (per pixel/channel), then re-normalized.
    Returns (fractions, accuracies%). t=0 is clean; t=1 is pure uniform noise (-> chance)."""
    dev = X.device
    m = mean.view(-1, 1, 1).to(dev); s = std.view(-1, 1, 1).to(dev)
    x01 = (X * s + m).clamp(0, 1)
    g = torch.Generator(device=dev).manual_seed(seed)
    noise = torch.rand(x01.shape, generator=g, device=dev)
    fracs = np.linspace(0.0, 1.0, n_points)
    accs = [accuracy(model, ((1.0 - t) * x01 + t * noise - m) / s, Y) for t in fracs]
    return fracs, np.array(accs) * 100.0


@torch.no_grad()
def _corrupt_batch(X, mean, std, name, lvl):
    """Apply noise corruption `name` at ladder level `lvl` (0..8) to a normalized batch; re-normalize."""
    m = mean.view(-1, 1, 1); s = std.view(-1, 1, 1)
    x01 = (X.cpu() * s + m).clamp(0, 1)
    ch = X.shape[1]
    base = np.random.RandomState(zlib.crc32(f"{name}|{lvl}".encode()) % (1 << 30))   # deterministic across processes (hash() is salted)
    out = []
    for img in x01:
        a = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        if a.shape[2] == 1:
            a = np.repeat(a, 3, 2)
        cc = C.apply(a, name, lvl, np.random.RandomState(base.randint(1 << 30)))
        t = torch.from_numpy(cc).float().div(255).permute(2, 0, 1)
        if ch == 1:
            t = t[:1]          # grayscale: noise the single channel once (as MNIST-C), not the mean of 3 independent draws
        out.append((t - m) / s)
    return torch.stack(out).to(X.device)


@torch.no_grad()
def noise_mca_curve(model, X, Y, mean, std, n_sev=9):
    """Accuracy at [clean, severity 1..n_sev], averaged over gaussian/shot/impulse (9-level ladders)."""
    sev = [accuracy(model, X, Y) * 100.0]
    for lvl in range(n_sev):
        accs = [accuracy(model, _corrupt_batch(X, mean, std, nm, lvl), Y) for nm in NOISE]
        sev.append(float(np.mean(accs)) * 100.0)
    return np.array(sev)
