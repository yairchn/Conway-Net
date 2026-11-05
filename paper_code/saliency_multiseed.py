"""Saliency on/off-object separation, multi-seed (paper protocol of saliency_onoff_separation.py: |d logit_y/dx|
channel-summed; object mask = luminance>0.1 (grayscale) or Otsu (colour), dilated by one pixel; sep = max(on/off,
off/on); images whose mask covers <3% or >97% of the frame are skipped). All 1000 first test images, true class y.
FCN x 4 grayscale datasets, CNN x 8 (grayscale + colour), with/without LI, 5 seeds each. Writes
diagnostics/saliency_multiseed.json. Run: PYTHONPATH=. python paper_code/saliency_multiseed.py
The paper reports this symmetric score only as region separability; the signed metrics are paper_code/saliency_signed_metrics.py.
"""
from conwaynet import REPO_ROOT
import json

import numpy as np
import torch
from scipy import ndimage

from conwaynet.config import load_config
from conwaynet.data import load_dataset
from conwaynet.models import build_model

ROOT = REPO_ROOT; DEV = "cuda"; N = 1000
MAP = json.load(open(f"{ROOT}/paper_runs/_paper_net_configs.json"))
FCN = {d: [f"fcn_{d}_tanh_ns_ki_d10"] + [f"fcn_{d}_tanh_ns_ki_d10_s{s}" for s in (43, 44, 45, 46)]
       for d in ("mnist", "fashionmnist", "kmnist", "emnist")}
CNN = {d: [f"cnn_{d}_perch"] + [f"cnn_{d}_perch_s{s}" for s in (1, 2, 3, 4)]
       for d in ("mnist", "fashionmnist", "kmnist", "emnist", "mnist_color", "fashionmnist_color", "kmnist_color", "emnist_color")}


def otsu(x):
    h, e = np.histogram(x.ravel(), 256, (0, 1)); p = h.astype(float) / max(h.sum(), 1)
    om = np.cumsum(p); mu = np.cumsum(p * ((e[:-1] + e[1:]) / 2)); sig = (mu[-1] * om - mu) ** 2 / (om * (1 - om) + 1e-12)
    return float((e[:-1] + e[1:])[np.nanargmax(sig)] / 2)


def mask_of(img01):
    lum, thr = (img01.mean(0), None) if img01.shape[0] > 1 else (img01[0], 0.1)
    return ndimage.binary_dilation(lum > (otsu(lum) if thr is None else thr), iterations=1)


def separation(m, X, Y, masks):
    out = []
    for i in range(0, len(X), 250):
        x = X[i:i + 250].clone().to(DEV).requires_grad_(True)
        m(x).gather(1, Y[i:i + 250].to(DEV)[:, None]).sum().backward()
        G = x.grad.abs().sum(1).cpu().numpy()
        for g, mk in zip(G, masks[i:i + 250]):
            f = mk.mean()
            if f < 0.03 or f > 0.97: continue
            on, off = max(float(g[mk].mean()), 1e-9), max(float(g[~mk].mean()), 1e-9); out.append(max(on / off, off / on))
    return float(np.mean(out))


res = {}
for arch, table in (("FCN", FCN), ("CNN", CNN)):
    for ds, runs in table.items():
        r = {"ca": [], "none": []}
        for run in runs:
            for k, mech in ((0, "ca"), (1, "none")):
                cfg = load_config(MAP[run][k])
                _, _, te, meta = load_dataset(cfg["data_path"], 500, cfg["data_name"], val_frac=cfg["val_frac"], split_seed=cfg["split_seed"],
                                              data_norms=cfg["data_norms"], output_path=f"{ROOT}/paper_runs/{run}", num_workers=2)
                X, Y = [], []
                for xb, yb in te:
                    X.append(xb); Y.append(yb)
                    if sum(len(x) for x in X) >= N: break
                X, Y = torch.cat(X)[:N], torch.cat(Y)[:N]
                mn = torch.tensor(meta["mean"]).view(1, -1, 1, 1); sd = torch.tensor(meta["std"]).view(1, -1, 1, 1)
                masks = [mask_of(im) for im in (X * sd + mn).clamp(0, 1).numpy()]
                m = build_model(cfg, meta); m.load_state_dict(torch.load(f"{ROOT}/paper_runs/{run}/model_{mech}.pth", map_location="cpu")); m.to(DEV).eval()
                r[mech].append(separation(m, X, Y, masks))
        res[f"{arch}|{ds}"] = {k: (float(np.mean(v)), float(np.std(v)), v) for k, v in r.items()}
        print(f"{arch} {ds:20s} with LI {np.mean(r['ca']):5.1f} ± {np.std(r['ca']):.1f}   without {np.mean(r['none']):5.1f} ± {np.std(r['none']):.1f}", flush=True)
json.dump(res, open(f"{ROOT}/diagnostics/saliency_multiseed.json", "w"), indent=1)
