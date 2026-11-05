"""Appendix A.6 controls on the PAPER CNN (MNIST): rule-free gates with the CA gate's wiring.

Arms (each trained once with the paper CNN recipe, configs/paper/cnn_mnist_perch_ca.json, seed 0,
via conwaynet.train -- the same loop as scripts/run.py):
  blur_<lam>  y = (1-lam) z + lam (2 m - 1): the CA gate's input->output wiring and the same
              occupancy m it reads (the per-channel image gate's normalised occupancy at the first
              gate, the channel-mean occupancy at the second), but no birth/survival rule
  sblur_<lam> shallow blur: y = (1-lam) z + lam (2 m(z) - 1), m the Moore mean of the output occupancy
              (z+1)/2 per channel (no input, no rule)
  dog_sm      published difference-of-Gaussians surround modulation (Hasani et al. 2019) applied to
              the conv output z only, at both gate sites
The with-LI and without-LI arms are the paper checkpoints (paper_runs/cnn_mnist_perch). Saves
paper_runs/cnn_mnist_perch_controls/model_<arm>.pth + results_<arm>.json (held-out clean accuracy,
noise accuracy: gaussian/shot/impulse x 5 severities on 2000 test images, as for the paper runs).
Run: PYTHONPATH=. python paper_code/cnn_controls.py <arm> [<arm> ...]
"""
from conwaynet import REPO_ROOT
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from conwaynet.config import load_config
from conwaynet.data import load_dataset
from conwaynet.eval import evaluate_mca, NOISE
from conwaynet.lateral_inhibition import moore_mean
from conwaynet.models import build_model
from conwaynet.train import train, evaluate

ROOT = REPO_ROOT
CFG = f"{ROOT}/configs/paper/cnn_mnist_perch_ca.json"
OUT = f"{ROOT}/paper_runs/cnn_mnist_perch_controls"


def _image_occupancy(x):
    """Occupancy the per-channel image gate reads: |m - median| / max per image and channel."""
    m = moore_mean((x + 1.0) / 2.0)
    flat = m.flatten(2)
    dev = (flat - flat.median(-1, keepdim=True).values).abs()
    return (dev / dev.amax(-1, keepdim=True).clamp_min(1e-6)).view_as(m).detach()


class BlurGate(nn.Module):
    def __init__(self, lam, image_site):
        super().__init__()
        self.lam, self.image_site = lam, image_site

    def forward(self, x, z):
        m = _image_occupancy(x) if self.image_site else moore_mean((x + 1.0) / 2.0)
        if m.shape[1] != z.shape[1]:
            m = m.mean(1, keepdim=True).expand(-1, z.shape[1], -1, -1)
        if m.shape[-2:] != z.shape[-2:]:
            m = F.interpolate(m, size=z.shape[-2:], mode="bilinear", align_corners=False)
        return (1.0 - self.lam) * z + self.lam * (2.0 * m - 1.0)


class ShallowBlurGate(nn.Module):
    def __init__(self, lam):
        super().__init__()
        self.lam = lam

    def forward(self, x, z):
        return (1.0 - self.lam) * z + self.lam * (2.0 * moore_mean((z + 1.0) / 2.0) - 1.0)


def dog_kernel(se=1.2, si=1.4, k=5):
    c = (k - 1) / 2
    yy, xx = np.mgrid[0:k, 0:k]
    ge = np.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2 * se ** 2)) / (2 * np.pi * se ** 2)
    gi = np.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2 * si ** 2)) / (2 * np.pi * si ** 2)
    d = ge - gi
    return torch.tensor(d / d[int(c), int(c)], dtype=torch.float32)


class DoGSMGate(nn.Module):
    def __init__(self, k=5):
        super().__init__()
        self.k = k
        self.register_buffer("dog", dog_kernel(k=k))

    def forward(self, x, z):
        c = z.shape[1]
        return F.conv2d(z, self.dog.to(z.dtype).view(1, 1, self.k, self.k).expand(c, 1, self.k, self.k),
                        padding=self.k // 2, groups=c)


def build(arm, meta, cfg):
    if arm in ("ca", "none"):
        c = load_config(CFG.replace("_ca.json", f"_{arm}.json"))
        m = build_model(c, meta)
        m.load_state_dict(torch.load(f"{ROOT}/paper_runs/cnn_mnist_perch/model_{arm}.pth", map_location="cpu"))
        return m
    m = build_model(cfg, meta)
    if arm.startswith("blur_"):
        lam = float(arm.split("_")[1])
        m.li1, m.li2 = BlurGate(lam, image_site=True), BlurGate(lam, image_site=False)
    elif arm.startswith("sblur_"):
        lam = float(arm.split("_")[1])
        m.li1, m.li2 = ShallowBlurGate(lam), ShallowBlurGate(lam)
    elif arm == "dog_sm":
        m.li1, m.li2 = DoGSMGate(), DoGSMGate()
    else:
        raise ValueError(arm)
    return m


def loaders(cfg, gen):
    return load_dataset(cfg["data_path"], cfg["batch_size"], cfg["data_name"], val_frac=cfg["val_frac"],
                        split_seed=cfg["split_seed"], data_norms=cfg["data_norms"], output_path=OUT,
                        num_workers=cfg["num_workers"], generator=gen)


def main(args):
    """args: arm names, optionally preceded by seed=K (K>0 trains seed K into <OUT>_sK; default: config seed, OUT)."""
    global OUT
    cfg = load_config(CFG)
    arms = [a for a in args if not a.startswith("seed=")]
    for a in args:
        if a.startswith("seed="):
            cfg["seed"] = int(a.split("=")[1]); OUT = f"{OUT}_s{cfg['seed']}"
    os.makedirs(OUT, exist_ok=True)
    for arm in arms:
        random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"]); torch.cuda.manual_seed_all(cfg["seed"])
        gen = torch.Generator(); gen.manual_seed(cfg["seed"])
        tr, va, te, meta = loaders(cfg, gen)
        model = build(arm, meta, cfg)
        best, last, bs = train(model, tr, va, cfg, meta, step_hook=None)
        if bs is not None:
            model.load_state_dict({k: v.to(cfg["device"]) for k, v in bs.items()})
        clean = evaluate(model, te, cfg["device"])
        res = evaluate_mca(model, cfg["data_path"], cfg["data_name"], meta["mean"], meta["std"], cfg["device"],
                           corruptions=NOISE, n_samples=cfg["mca_n_samples"], batch_size=cfg["batch_size"])
        torch.save(model.state_dict(), f"{OUT}/model_{arm}.pth")
        json.dump(dict(arm=arm, config=CFG, best_val_acc=best, last_val_acc=last, test_clean_acc=clean,
                       noise_mCA=res["noise_mCA"], per_corruption={k: res[k] for k in NOISE}),
                  open(f"{OUT}/results_{arm}.json", "w"), indent=2)
        print(f"{arm}: clean {clean:.4f}  noise {res['noise_mCA']:.4f}  -> {OUT}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
