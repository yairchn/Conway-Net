"""Ergodicity check of the FCN's hidden maps (depth = time, independently trained seeds = ensemble), MNIST, 5 seeds per
arm (paper networks, seeds 42-46), first 500 test images, deep layers 6-10 (the quasi-stationary range).
Per image (all maps flattened to the 784 units of the 28x28 grid):
  time average      T_s   = mean over layers 6-10 of seed s's maps            (one realisation, averaged in time)
  ensemble average  E_l^-s = mean over the other four seeds at layer l          (leave-one-out, no trivial overlap)
  r_time_ens   = corr(T_s, E_l^-s)          does one network's depth average match the ensemble average?
  r_snap_ens   = corr(A_{s,l}, E_l^-s)      a single snapshot against the ensemble (no time averaging)
  r_half       = corr(E over seeds {a,b}, E over seeds {c,d}) at layer l   reference: ensemble vs ensemble
  var_time / var_ens: mean squared fluctuation of A_{s,l} around T_s, and around the ensemble mean E_l (all 5 seeds)
Means over images, seeds, layers. Writes diagnostics/fcn_ergodicity.json.
Run: PYTHONPATH=. python paper_code/fcn_ergodicity.py
"""
from conwaynet import REPO_ROOT
import itertools
import json

import numpy as np
import torch

from conwaynet.config import load_config
from conwaynet.data import load_dataset
from conwaynet.models import build_model

ROOT = REPO_ROOT; DEV = "cuda"; N = 500; DEEP = list(range(5, 10))     # layers 6-10 (0-based 5..9)
MAP = json.load(open(f"{ROOT}/paper_runs/_paper_net_configs.json"))
RUNS = ["fcn_mnist_tanh_ns_ki_d10"] + [f"fcn_mnist_tanh_ns_ki_d10_s{s}" for s in (43, 44, 45, 46)]


@torch.no_grad()
def hidden(m, X):
    cur = X.view(X.size(0), -1); outs = []
    for i, lin in enumerate(m.linears):
        t = lin(cur); t = m.norms[i](t) if m.norms is not None else t; z = torch.tanh(t)
        if m.gate is not None: z = m.gate(cur, z)
        cur = z; outs.append(cur.cpu().numpy())
    return np.stack(outs)                                              # (layers, N, 784)


def rowcorr(a, b):
    a = a - a.mean(-1, keepdims=True); b = b - b.mean(-1, keepdims=True)
    return (a * b).sum(-1) / np.sqrt((a * a).sum(-1) * (b * b).sum(-1) + 1e-12)


cfg = load_config(MAP[RUNS[0]][0])
_, _, te, meta = load_dataset(cfg["data_path"], 500, cfg["data_name"], val_frac=cfg["val_frac"], split_seed=cfg["split_seed"],
                              data_norms=cfg["data_norms"], output_path=None, num_workers=2)
X = next(iter(te))[0][:N].to(DEV)
out = {}
for k, mech in ((0, "ca"), (1, "none")):
    A = []
    for run in RUNS:
        m = build_model(load_config(MAP[run][k]), meta); m.load_state_dict(torch.load(f"{ROOT}/paper_runs/{run}/model_{mech}.pth", map_location="cpu")); m.to(DEV).eval()
        A.append(hidden(m, X)[DEEP])
    A = np.stack(A)                                                    # (seeds, layers, N, 784)
    S, L = A.shape[:2]; T = A.mean(1)                                  # time average per seed (S, N, 784)
    r_te, r_se, r_half = [], [], []
    for s in range(S):
        Eo = A[[t for t in range(S) if t != s]].mean(0)                # leave-one-out ensemble mean per layer (L, N, 784)
        for l in range(L):
            r_te.append(rowcorr(T[s], Eo[l]).mean()); r_se.append(rowcorr(A[s, l], Eo[l]).mean())
    for (a, b), (c, d) in [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)), ((1, 4), (2, 3)), ((0, 4), (1, 2))]:
        for l in range(L):
            r_half.append(rowcorr(A[[a, b], l].mean(0), A[[c, d], l].mean(0)).mean())
    var_time = float(((A - T[:, None]) ** 2).mean()); var_ens = float(((A - A.mean(0, keepdims=True)) ** 2).mean())
    out[mech] = dict(r_time_ens=float(np.mean(r_te)), r_snap_ens=float(np.mean(r_se)), r_half=float(np.mean(r_half)),
                     var_time=var_time, var_ens=var_ens, var_total=float(A.var()))
    o = out[mech]
    print(f"{'with LI' if mech == 'ca' else 'plain  '}: corr(time avg of one network, ensemble avg) {o['r_time_ens']:.3f} | "
          f"single snapshot vs ensemble {o['r_snap_ens']:.3f} | ensemble half vs half {o['r_half']:.3f} | "
          f"fluctuation variance around time avg {o['var_time']:.4f}, around ensemble avg {o['var_ens']:.4f} (total {o['var_total']:.4f})", flush=True)
json.dump(out, open(f"{ROOT}/diagnostics/fcn_ergodicity.json", "w"), indent=1)
