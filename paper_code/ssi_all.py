"""Reviewer P6: recompute the scale-separation index with the one definition of scale_separation_index.py (DC bin
centred, conjugate-symmetric masks, common horizon of 9 lags, first 1000 test images, censoring flagged) for every
FCN network the paper reports: B5/S3 and plain at depth 10 (5 seeds), B5/S3 at depth 100 (one network; d100 plain was
not trained), the nine other rules of the sweep (5 seeds each), the FCN controls (paper_runs/fcn_mnist_controls) and
the gate-route cut (runs/gate_detached). For the depth-10 networks also the former computation (masks centred at
13.5, the half-pixel bug of fcn_scale_separation.py / fcn_depth_diagnostics.py / fcn_controls_metrics.py), to
measure what the correction changes. Writes diagnostics/ssi_all.json.
Run: PYTHONPATH=. python paper_code/ssi_all.py
"""
from conwaynet import REPO_ROOT
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, f"{REPO_ROOT}/paper_code")
import fcn_controls as FC
import scale_separation_index as SS
from conwaynet.config import load_config
from conwaynet.data import load_dataset
from conwaynet.models import build_model

ROOT = REPO_ROOT; DEV = "cuda"; HW = 28
MAP = json.load(open(f"{ROOT}/paper_runs/_paper_net_configs.json"))
SEEDS = (42, 43, 44, 45, 46)
RULES = ["B5S3", "B5S23", "B3S3", "B3S23", "B34S34", "B36S23", "B3678S34678", "B45678S2345", "B2S", "B1357S1357"]


def runs(tag):
    if tag in ("B5S3", "none"):
        return {k: ("fcn_mnist_tanh_ns_ki_d10" if k == 42 else f"fcn_mnist_tanh_ns_ki_d10_s{k}") for k in SEEDS}
    return {k: f"fcn_mnist_tanh_ns_ki_d10_{tag}_s{k}" for k in SEEDS}


@torch.no_grad()
def hidden_maps(m, X):
    cur = X.view(X.size(0), -1); maps = []
    for i, lin in enumerate(m.linears):
        t = lin(cur); t = m.norms[i](t) if m.norms is not None else t; z = torch.tanh(t)
        if m.gate is not None:
            z = m.gate(cur, z)
        cur = z; maps.append(z.view(z.size(0), HW, HW))
    return torch.stack(maps)


@torch.no_grad()
def ssi_former(maps):
    """The former computation (masks centred at (HW - 1)/2 = 13.5, float32), for comparison only."""
    L = maps.shape[0]; F = torch.fft.fftshift(torch.fft.fft2(maps), dim=(-2, -1))
    yy, xx = np.meshgrid(np.arange(HW), np.arange(HW), indexing="ij"); r = np.sqrt((yy - (HW - 1) / 2) ** 2 + (xx - (HW - 1) / 2) ** 2)
    taus = []
    for lo, hi in zip(SS.BAND_EDGES[:-1], SS.BAND_EDGES[1:]):
        mk = torch.tensor(((r >= lo) & (r < hi)).astype(np.float32), device=maps.device)
        Mb = torch.fft.ifft2(torch.fft.ifftshift(F * mk, dim=(-2, -1))).real
        v = Mb.view(L, Mb.shape[1], -1); v = v - v.mean(-1, keepdim=True); vn = v / (v.norm(dim=-1, keepdim=True) + 1e-9)
        C = [float((vn[:L - k] * vn[k:]).sum(-1).mean()) for k in range(0, min(9, L - 1) + 1)]
        s = 0.0
        for k in range(1, len(C)):
            if C[k] <= 0: break
            s += C[k] / C[0]
        taus.append(0.5 + s)
    return taus[0] / taus[-1]


def test_images(cfg, run_dir):
    _, _, te, _ = load_dataset(cfg["data_path"], 500, cfg["data_name"], val_frac=cfg["val_frac"], split_seed=cfg["split_seed"],
                               data_norms=cfg["data_norms"], output_path=run_dir, num_workers=2)
    X = []
    for xb, _ in te:
        X.append(xb)
        if sum(len(x) for x in X) >= SS.N_IMAGES: break
    return torch.cat(X)[:SS.N_IMAGES].to(DEV)


def score(m, X, former=False):
    maps = hidden_maps(m.to(DEV).eval(), X); r = SS.scale_separation(maps)
    if former:
        r["ssi_former_13p5"] = ssi_former(maps.float())
    return r


res = {}
for tag in RULES + ["none"]:
    for seed, run in runs(tag).items():
        cfg = load_config(MAP[run][1 if tag == "none" else 0]) if run in MAP else load_config(f"{ROOT}/paper_runs/{run}/config_ca.json")
        mech = "none" if tag == "none" else "ca"
        X = test_images(cfg, f"{ROOT}/paper_runs/{run}")
        meta = dict(channels=1, image_size=HW, num_classes=10)
        m = build_model(cfg, meta); m.load_state_dict(torch.load(f"{ROOT}/paper_runs/{run}/model_{mech}.pth", map_location="cpu"))
        res[f"d10|{tag}|{seed}"] = r = score(m, X, former=tag in ("B5S3", "none"))
        print(f"d10 {tag:12s} seed {seed}  SSI {r['ssi']:5.2f}  tau coarse {r['tau'][0]:.2f}{' (censored)' if r['censored'][0] else ''} "
              f"fine {r['tau'][-1]:.2f}  C_coarse(9) {r['C'][0][9]:.3f}  C_fine(1) {r['C'][-1][1]:+.3f}"
              + (f"  former {r['ssi_former_13p5']:.2f}" if "ssi_former_13p5" in r else ""), flush=True)
cfg = load_config(f"{ROOT}/configs/paper/tanh/fcn_mnist_tanh_ns_ki_d100_ca.json"); run = f"{ROOT}/paper_runs/fcn_mnist_tanh_ns_ki_d100"
X = test_images(cfg, run); m = build_model(cfg, dict(channels=1, image_size=HW, num_classes=10))
m.load_state_dict(torch.load(f"{run}/model_ca.pth", map_location="cpu"))
res["d100|B5S3|42"] = r = score(m, X)
print(f"d100 B5S3  SSI {r['ssi']:5.2f}  tau {[round(t, 2) for t in r['tau']]}  censored {r['censored']}", flush=True)
del m; torch.cuda.empty_cache()

c0 = load_config(FC.CFG.format("ca")); X = test_images(c0, f"{ROOT}/paper_runs/fcn_mnist_tanh_ns_ki_d10")
meta = dict(channels=1, image_size=HW, num_classes=10, mean=[0.5], std=[0.5])
for d in sorted(os.listdir(FC.OUT)):
    p = f"{FC.OUT}/{d}/model.pth"
    if not os.path.exists(p):
        continue
    arm, seed = d.rsplit("_s", 1)
    m, _ = FC.build(arm, int(seed), meta); m.load_state_dict(torch.load(p, map_location="cpu"))
    res[f"controls|{arm}|{seed}"] = r = score(m, X)
    print(f"controls {arm:12s} seed {seed}  SSI {r['ssi']:5.2f}", flush=True)
cfg = load_config(f"{ROOT}/configs/paper/tanh/fcn_mnist_tanh_ns_ki_ca.json")
for arm in ("full", "detached"):
    m = build_model(cfg, dict(channels=1, image_size=HW, num_classes=10))
    m.load_state_dict(torch.load(f"{ROOT}/runs/gate_detached/fcn_mnist_{arm}/model_ca.pth", map_location="cpu"))
    res[f"gate_routes|{arm}|42"] = r = score(m, X)
    print(f"gate routes {arm:9s}  SSI {r['ssi']:5.2f}", flush=True)
json.dump(res, open(f"{ROOT}/diagnostics/ssi_all.json", "w"), indent=1)
print("WROTE diagnostics/ssi_all.json")
