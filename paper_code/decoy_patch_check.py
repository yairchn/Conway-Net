"""Decoy-patch check (reviewer: 'less attribution on the patch yet more flips at 14x14'), paper CNN on MNIST, 5 seeds.

Same protocol as decoy_patch.py (bright square of side s at a seeded random location, on correctly classified test
images; integrated gradients, black baseline, 32 steps, for the true class), plus:
  overlap        fraction of the patch's pixels on the dilated digit mask (pixel > 0.1, dilated by one pixel);
                 flips are split by overlap: off-digit (0), partial (0-50%), less than half on the digit (< 50%), mostly on (>= 50%)
  attribution    (a) magnitude share: sum |IG| on patch / sum |IG| (as in the paper figure)
                 (b) signed shares: positive IG (supports the class) on patch / all positive IG, same for negative IG
                 (c) top-k share: of the k = s*s most-attributed pixels (|IG|), the fraction inside the patch
                     (chance = patch area / image area)
Run: PYTHONPATH=. python paper_code/decoy_patch_check.py
"""
from conwaynet import REPO_ROOT
import json

import numpy as np
import torch
from scipy import ndimage

from conwaynet.config import load_config
from conwaynet.data import load_dataset
from conwaynet.models import build_model

ROOT = REPO_ROOT; DEV = "cuda"; N = 1000; SIZES = [2, 4, 6, 8, 10]
MAP = json.load(open(f"{ROOT}/paper_runs/_paper_net_configs.json"))
RUNS = ["cnn_mnist_perch"] + [f"cnn_mnist_perch_s{k}" for k in (1, 2, 3, 4)]


def ig(m, x, t, steps=32):
    base = torch.full_like(x, -1.0); tot = torch.zeros_like(x)
    for a in torch.linspace(1.0 / steps, 1.0, steps):
        xi = (base + a * (x - base)).requires_grad_(True)
        g, = torch.autograd.grad(m(xi).gather(1, t[:, None]).sum(), xi); tot += g
    return ((x - base) * tot / steps).sum(1)                     # signed, (B,H,W)


cfg0 = load_config(MAP[RUNS[0]][0])
_, _, te, meta = load_dataset(cfg0["data_path"], 500, cfg0["data_name"], val_frac=cfg0["val_frac"], split_seed=cfg0["split_seed"],
                              data_norms=cfg0["data_norms"], output_path=None, num_workers=2)
X0, Y0 = [], []
for xb, yb in te:
    X0.append(xb); Y0.append(yb)
    if sum(len(x) for x in X0) >= N: break
X0, Y0 = torch.cat(X0)[:N], torch.cat(Y0)[:N]
DIG = torch.from_numpy(np.stack([ndimage.binary_dilation(i > 0.1, iterations=1) for i in (X0[:, 0] * 0.5 + 0.5).numpy()]))

out = {}
for mech in ("ca", "none"):
    per_seed = []
    for run in RUNS:
        m = build_model(load_config(MAP[run][0 if mech == "ca" else 1]), meta)
        m.load_state_dict(torch.load(f"{ROOT}/paper_runs/{run}/model_{mech}.pth", map_location="cpu")); m.to(DEV).eval()
        with torch.no_grad(): keep = (m(X0.to(DEV)).argmax(1).cpu() == Y0)
        X, Y, D = X0[keep].to(DEV), Y0[keep].to(DEV), DIG[keep].to(DEV)
        H, W = X.shape[-2:]; rows = {}
        for s in SIZES:
            rng = np.random.RandomState(1000 + s)
            ii = rng.randint(0, H - s + 1, len(X)); jj = rng.randint(0, W - s + 1, len(X))
            Xp = X.clone(); M = torch.zeros(len(X), H, W, dtype=torch.bool, device=DEV)
            for k in range(len(X)):
                Xp[k, :, ii[k]:ii[k] + s, jj[k]:jj[k] + s] = 1.0; M[k, ii[k]:ii[k] + s, jj[k]:jj[k] + s] = True
            ov = (M & D).flatten(1).sum(1).float() / (s * s)
            with torch.no_grad(): flip = (m(Xp).argmax(1) != Y).float()
            A = torch.cat([ig(m, Xp[b:b + 200], Y[b:b + 200]) for b in range(0, len(Xp), 200)])
            Aa, Mf = A.abs().flatten(1), M.flatten(1)
            mag = (Aa * Mf).sum(1) / Aa.sum(1).clamp_min(1e-12)
            P, Ng = A.clamp(min=0).flatten(1), (-A).clamp(min=0).flatten(1)
            pos = (P * Mf).sum(1) / P.sum(1).clamp_min(1e-12); neg = (Ng * Mf).sum(1) / Ng.sum(1).clamp_min(1e-12)
            top = Aa.topk(s * s, dim=1).indices; topk = Mf.gather(1, top).float().mean(1)
            bins = {"off": ov == 0, "partial": (ov > 0) & (ov < 0.5), "lt50": ov < 0.5, "on": ov >= 0.5}
            rows[s] = dict(flip=float(flip.mean()), mag=float(mag.mean()), pos=float(pos.mean()), neg=float(neg.mean()),
                           topk=float(topk.mean()), **{f"flip_{b}": float(flip[v].mean()) if v.any() else float("nan") for b, v in bins.items()},
                           **{f"frac_{b}": float(v.float().mean()) for b, v in bins.items()})
        per_seed.append(rows); del m; torch.cuda.empty_cache()
    out[mech] = {s: {k: (float(np.nanmean([r[s][k] for r in per_seed])), float(np.nanstd([r[s][k] for r in per_seed])))
                     for k in per_seed[0][s]} for s in SIZES}

print("patch | flip % (with / plain) | flips by patch-on-digit: off | partial | >=50% (with / plain) | "
      "IG magnitude share | IG positive share | IG negative share | top-k pixels in patch  (chance)")
for s in SIZES:
    c, n = out["ca"][s], out["none"][s]; f = lambda k: f"{c[k][0]*100:5.1f}/{n[k][0]*100:5.1f}"
    print(f"{s:2d}px | {f('flip')} (±{c['flip'][1]*100:.1f}/{n['flip'][1]*100:.1f}) | {f('flip_off')} | {f('flip_partial')} | {f('flip_on')} "
          f"[share of patches: {c['frac_off'][0]*100:.0f}/{c['frac_partial'][0]*100:.0f}/{c['frac_on'][0]*100:.0f}%] | "
          f"{f('mag')} | {f('pos')} | {f('neg')} | {f('topk')}  ({s*s/784*100:.1f})")
json.dump({m: {str(s): v for s, v in d.items()} for m, d in out.items()}, open(f"{ROOT}/diagnostics/decoy_patch_check.json", "w"), indent=1)
