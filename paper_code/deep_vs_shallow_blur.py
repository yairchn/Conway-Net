"""Blur controls of the saliency appendix (output-blur rows) and the FCN saliency/noise numbers there: deep blur y = (1-lam) z + lam pi(m(x)) vs shallow blur y = (1-lam) z + lam pi(m(z)),
MNIST, seed 0. Scores each saved network on held-out clean accuracy, noise accuracy (conwaynet.eval.evaluate_mca:
gaussian/shot/impulse x 5 standard severities, 2000 test images) and saliency on/off-digit separation over the first
1000 test images (as paper_code/saliency_multiseed.py). Activation coherence (LFPR etc.) comes from
paper_code/fcn_controls_metrics.py, which picks up the sblur_* FCN runs on its own.
  FCN: paper_runs/fcn_mnist_controls/<arm>_s0   (paper_code/fcn_controls.py; localmix = sblur_0.5)
  CNN: paper_runs/cnn_mnist_perch (ca, none) and paper_runs/cnn_mnist_perch_controls (blur_*, sblur_*)
Writes diagnostics/deep_vs_shallow_blur.json.
Run: PYTHONPATH=. python paper_code/deep_vs_shallow_blur.py
The paper reports the symmetric score as region separability; the signed log-ratio is paper_code/saliency_signed_controls.py.
"""
from conwaynet import REPO_ROOT
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, f"{REPO_ROOT}/paper_code")
import cnn_controls as CC
import fcn_controls as FC
_src = open(f"{REPO_ROOT}/paper_code/saliency_multiseed.py").read()
exec(_src[:_src.index("\nres = {}")])      # mask_of, separation
from conwaynet.config import load_config
from conwaynet.data import load_dataset
from conwaynet.eval import evaluate_mca, NOISE
from conwaynet.train import evaluate

ROOT = REPO_ROOT; DEV = "cuda"


def first(te, n):
    X, Y = [], []
    for xb, yb in te:
        X.append(xb); Y.append(yb)
        if sum(len(x) for x in X) >= n: break
    return torch.cat(X)[:n], torch.cat(Y)[:n]


def score(m, cfg, meta, te, X, Y, masks):
    m.to(DEV).eval()
    noise = evaluate_mca(m, cfg["data_path"], cfg["data_name"], meta["mean"], meta["std"], DEV, corruptions=NOISE,
                         n_samples=cfg["mca_n_samples"], batch_size=cfg["batch_size"])["noise_mCA"]
    return dict(clean=evaluate(m, te, DEV), noise=noise, sep=separation(m, X, Y, masks))


res = {"FCN": {}, "CNN": {}}
# ---------------------------------------------------------------- FCN
c = load_config(FC.CFG.format("ca"))
_, _, te, meta = load_dataset(c["data_path"], 500, c["data_name"], val_frac=c["val_frac"], split_seed=c["split_seed"],
                              data_norms=c["data_norms"], output_path=None, num_workers=2)
X, Y = first(te, 1000); masks = [mask_of(im) for im in (X * 0.5 + 0.5).clamp(0, 1).numpy()]
FCN_ARMS = ["ca", "none"] + [f"blur_{l}" for l in ("0.5", "0.75", "0.9", "0.95", "1.0")] + \
           ["localmix"] + [f"sblur_{l}" for l in ("0.75", "0.9", "0.95", "1.0")]
for arm in FCN_ARMS:
    p = f"{FC.OUT}/{arm}_s0/model.pth"
    if not os.path.exists(p):
        print("missing", p); continue
    m, cfg = FC.build(arm, 0, meta); m.load_state_dict(torch.load(p, map_location="cpu"))
    res["FCN"][arm] = r = score(m, cfg, meta, te, X, Y, masks)
    print(f"FCN {arm:11s} clean {r['clean']*100:5.1f}  noise {r['noise']*100:5.1f}  sep {r['sep']:5.2f}", flush=True)
# ---------------------------------------------------------------- CNN
cfg = load_config(CC.CFG)
_, _, te, meta = load_dataset(cfg["data_path"], cfg["batch_size"], cfg["data_name"], val_frac=cfg["val_frac"], split_seed=cfg["split_seed"],
                              data_norms=cfg["data_norms"], output_path=f"{ROOT}/paper_runs/cnn_mnist_perch", num_workers=2)
X, Y = first(te, 1000); masks = [mask_of(im) for im in (X * 0.5 + 0.5).clamp(0, 1).numpy()]
for arm in ["ca", "none", "blur_0.5", "blur_0.9", "blur_0.95", "sblur_0.5", "sblur_0.9", "sblur_0.95"]:
    if arm not in ("ca", "none") and not os.path.exists(f"{CC.OUT}/model_{arm}.pth"):
        print("missing", arm); continue
    m = CC.build(arm, meta, cfg)
    if arm not in ("ca", "none"):
        m.load_state_dict(torch.load(f"{CC.OUT}/model_{arm}.pth", map_location="cpu"))
    res["CNN"][arm] = r = score(m, cfg, meta, te, X, Y, masks)
    print(f"CNN {arm:11s} clean {r['clean']*100:5.1f}  noise {r['noise']*100:5.1f}  sep {r['sep']:5.2f}", flush=True)
json.dump(res, open(f"{ROOT}/diagnostics/deep_vs_shallow_blur.json", "w"), indent=1)
print("WROTE diagnostics/deep_vs_shallow_blur.json")
