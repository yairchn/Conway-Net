"""Reviewer P2: signed, per-image saliency localisation metrics (the symmetric score max(on/off, off/on) cannot tell
object from background reliance). Same networks, images, saliency and mask as saliency_multiseed.py:
|d logit_t / dx| channel-summed; object mask = luminance > 0.1 (grayscale) or Otsu (colour), dilated by one pixel;
first 1000 test images of each run's saved split. Per image, with on / off = mean saliency on / off the mask and
f = mask area / image area:
  log_ratio  = log10(on / off)                       > 0 favours the object, < 0 the background
  obj_mass   = (sum on-mask / sum all) / f          area-normalised object attribution mass; 1 = uniform
  favours    = on > off
  region_sep = max(on/off, off/on)                   the paper's former score, renamed region separability
Exclusions, counted and reported: mask covers < 3% or > 97% of the image; vanishing saliency, max |d logit_t/dx| below
ZERO_GRAD = 1e-8 (float32 rounding level; full gradients are ~1e-2), for which no metric is defined; on or off exactly
zero (log_ratio undefined: excluded from log_ratio, kept in favours and obj_mass).
Groups: all images, correctly classified, misclassified. Target t = true class (as for the paper's numbers) and, for
the misclassified, also the predicted class. Networks: FCN x 4 grayscale, CNN x 8 (grayscale + colour), 5 seeds,
with / without lateral inhibition; WRN EMNIST (1 seed); MNIST FCN/CNN also untrained and with every gate route
detached (trunk only). Writes diagnostics/saliency_signed_metrics.json and diagnostics/saliency_signed_perimage.npz.
Run: PYTHONPATH=. python paper_code/saliency_signed_metrics.py
"""
from conwaynet import REPO_ROOT
import importlib
import json
import random

import numpy as np
import torch
from scipy import ndimage

LIM = importlib.import_module("conwaynet.lateral_inhibition")
from conwaynet.config import load_config
from conwaynet.data import load_dataset
from conwaynet.models import build_model

ROOT = REPO_ROOT; DEV = "cuda"; N = 1000; ZERO_GRAD = 1e-8
MAP = json.load(open(f"{ROOT}/paper_runs/_paper_net_configs.json"))
FCN = {d: [f"fcn_{d}_tanh_ns_ki_d10"] + [f"fcn_{d}_tanh_ns_ki_d10_s{s}" for s in (43, 44, 45, 46)]
       for d in ("mnist", "fashionmnist", "kmnist", "emnist")}
CNN = {d: [f"cnn_{d}_perch"] + [f"cnn_{d}_perch_s{s}" for s in (1, 2, 3, 4)]
       for d in ("mnist", "fashionmnist", "kmnist", "emnist", "mnist_color", "fashionmnist_color", "kmnist_color", "emnist_color")}
WRN = {"emnist": ["wrn_emnist"]}

_orig_forward = LIM.LateralInhibitionCA.forward
DETACH = {"all": False}


def _patched(self, x, z):
    return _orig_forward(self, x.detach() if DETACH["all"] else x, z)


LIM.LateralInhibitionCA.forward = _patched


def otsu(x):
    h, e = np.histogram(x.ravel(), 256, (0, 1)); p = h.astype(float) / max(h.sum(), 1)
    om = np.cumsum(p); mu = np.cumsum(p * ((e[:-1] + e[1:]) / 2)); sig = (mu[-1] * om - mu) ** 2 / (om * (1 - om) + 1e-12)
    return float((e[:-1] + e[1:])[np.nanargmax(sig)] / 2)


def mask_of(img01):
    lum, thr = (img01.mean(0), None) if img01.shape[0] > 1 else (img01[0], 0.1)
    return ndimage.binary_dilation(lum > (otsu(lum) if thr is None else thr), iterations=1)


def logits(m, x):
    o = m(x)
    return o[0] if isinstance(o, tuple) else o


def saliency(m, X, T):
    out = []
    for i in range(0, len(X), 250):
        x = X[i:i + 250].clone().to(DEV).requires_grad_(True)
        logits(m, x).gather(1, T[i:i + 250].to(DEV)[:, None]).sum().backward()
        out.append(x.grad.abs().sum(1).cpu().numpy())
    return np.concatenate(out)


def per_image(S, masks):
    """on, off, on-mass share, mask fraction per image."""
    on = np.array([g[mk].mean() for g, mk in zip(S, masks)])
    off = np.array([g[~mk].mean() for g, mk in zip(S, masks)])
    tot = S.reshape(len(S), -1).sum(1); gmax = S.reshape(len(S), -1).max(1)
    share = np.array([g[mk].sum() for g, mk in zip(S, masks)]) / np.where(tot > 0, tot, 1)
    f = np.array([mk.mean() for mk in masks])
    return dict(on=on, off=off, tot=tot, gmax=gmax, share=share, f=f, npix=S[0].size)


def summarise(p, idx):
    on, off, tot, gmax, share, f = (p[k][idx] for k in ("on", "off", "tot", "gmax", "share", "f"))
    n0 = len(idx)
    keep_mask = (f >= 0.03) & (f <= 0.97); keep_grad = gmax >= ZERO_GRAD; keep = keep_mask & keep_grad
    lr_ok = keep & (on > 0) & (off > 0)
    lr = np.log10(on[lr_ok] / off[lr_ok])
    mass = share[keep] / f[keep]
    q = np.percentile(lr, [10, 25, 50, 75, 90]) if len(lr) else [np.nan] * 5
    return dict(mean_abs_grad=float(tot.mean() / p["npix"]), n=n0, excl_mask=int((~keep_mask).sum()), excl_zero_grad=int((keep_mask & ~keep_grad).sum()),
                excl_logratio_one_side_zero=int((keep & ~lr_ok).sum()), n_used=int(keep.sum()),
                log_ratio_mean=float(lr.mean()) if len(lr) else np.nan, log_ratio_q10_25_50_75_90=[float(v) for v in q],
                obj_mass_mean=float(mass.mean()) if len(mass) else np.nan,
                obj_mass_median=float(np.median(mass)) if len(mass) else np.nan,
                frac_favours_object=float((on[keep] > off[keep]).mean()) if keep.any() else np.nan,
                region_sep_mean=float(np.mean(np.maximum(on[lr_ok] / off[lr_ok], off[lr_ok] / on[lr_ok]))) if len(lr) else np.nan)


def evaluate(m, X, Y, masks, tag, store):
    with torch.no_grad():
        P = torch.cat([logits(m, X[i:i + 250].to(DEV)).argmax(1).cpu() for i in range(0, len(X), 250)])
    correct = (P == Y).numpy()
    pt = per_image(saliency(m, X, Y), masks)
    out = {"acc": float(correct.mean()), "true_class": {g: summarise(pt, np.flatnonzero(sel)) for g, sel in
           (("all", np.ones(len(Y), bool)), ("correct", correct), ("misclassified", ~correct))}}
    wrong = np.flatnonzero(~correct)
    if len(wrong):
        pp = per_image(saliency(m, X[wrong], P[wrong]), [masks[k] for k in wrong])
        out["predicted_class_misclassified"] = summarise(pp, np.arange(len(wrong)))
    for k in ("on", "off", "tot", "gmax", "share", "f"):
        store[f"{tag}|{k}"] = pt[k].astype(np.float32)
    store[f"{tag}|correct"] = correct
    return out


def load_images(cfg, run):
    _, _, te, meta = load_dataset(cfg["data_path"], 500, cfg["data_name"], val_frac=cfg["val_frac"], split_seed=cfg["split_seed"],
                                  data_norms=cfg["data_norms"], output_path=f"{ROOT}/paper_runs/{run}", num_workers=2)
    X, Y = [], []
    for xb, yb in te:
        X.append(xb); Y.append(yb)
        if sum(len(x) for x in X) >= N: break
    X, Y = torch.cat(X)[:N], torch.cat(Y)[:N]
    mn = torch.tensor(meta["mean"]).view(1, -1, 1, 1); sd = torch.tensor(meta["std"]).view(1, -1, 1, 1)
    return X, Y, [mask_of(im) for im in (X * sd + mn).clamp(0, 1).numpy()], meta


res, store = {}, {}
for arch, table in (("FCN", FCN), ("CNN", CNN), ("WRN", WRN)):
    for ds, runs in table.items():
        for run in runs:
            for k, mech in ((0, "ca"), (1, "none")):
                cfg = load_config(MAP[run][k]); X, Y, masks, meta = load_images(cfg, run)
                variants = ["trained"] + (["untrained"] if ds == "mnist" and arch != "WRN" else [])
                for v in variants:
                    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"]); torch.cuda.manual_seed_all(cfg["seed"])
                    m = build_model(cfg, meta)
                    if v == "trained":
                        m.load_state_dict(torch.load(f"{ROOT}/paper_runs/{run}/model_{mech}.pth", map_location="cpu"))
                    m.to(DEV).eval()
                    res[f"{arch}|{ds}|{mech}|{v}|{run}"] = evaluate(m, X, Y, masks, f"{arch}|{ds}|{mech}|{v}|{run}", store)
                    if v == "trained" and mech == "ca" and ds == "mnist" and arch != "WRN":
                        DETACH["all"] = True
                        res[f"{arch}|{ds}|{mech}|trunk_only|{run}"] = evaluate(m, X, Y, masks, f"{arch}|{ds}|{mech}|trunk_only|{run}", store)
                        DETACH["all"] = False
            a = res[f"{arch}|{ds}|ca|trained|{run}"]["true_class"]["all"]; b = res[f"{arch}|{ds}|none|trained|{run}"]["true_class"]["all"]
            print(f"{arch} {ds:20s} {run:36s} log10 ratio {a['log_ratio_mean']:+.2f} / {b['log_ratio_mean']:+.2f}  "
                  f"obj mass {a['obj_mass_mean']:.2f} / {b['obj_mass_mean']:.2f}  favours obj {a['frac_favours_object']:.2f} / {b['frac_favours_object']:.2f}", flush=True)
json.dump(res, open(f"{ROOT}/diagnostics/saliency_signed_metrics.json", "w"), indent=1)
np.savez_compressed(f"{ROOT}/diagnostics/saliency_signed_perimage.npz", **store)
print("WROTE diagnostics/saliency_signed_metrics.json, saliency_signed_perimage.npz")
