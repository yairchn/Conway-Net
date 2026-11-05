"""Where each network's co-activation graph sits relative to its own percolation threshold, without choosing a
degree: r_c, the correlation of the edge at which the giant component reaches half the live units (edges added
strongest first, as in percolation_epochs.py). Also, on the graph of strong correlations (r > R_STRONG):
  - mean degree and its ratio to k_c (ratio > 1 iff r_c > R_STRONG)
  - fraction of its edges between units whose pixels are within 2 px (FCN: pixel = index % 784)
  - site percolation: f_c, the fraction of live units removed at random at which the giant falls below half the
    remaining units (N_SITE draws per fraction)
Composition control for the plain FCN: the same quantities on the fixed set of units live at the last epoch,
at every epoch in EPOCHS_FIXED (the ratio falls with training either because survivors lose strong partners
to unit death, or because the survivors' own correlations change).
Writes diagnostics/percolation_operating_point.json.
Run: PYTHONPATH=. python paper_code/percolation_operating_point.py
"""
from conwaynet import REPO_ROOT
import json
import sys

import numpy as np
import torch

sys.path.insert(0, f"{REPO_ROOT}/paper_code")
from percolation_epochs import hidden, corr, percolate, summary, N_NODES, N_IMG, SD_MIN, ROOT, DEV  # noqa: E402
from percolation_blur_snapshots import build_blur                                                  # noqa: E402
from conwaynet.config import load_config                                                           # noqa: E402
from conwaynet.data import load_dataset                                                            # noqa: E402
from conwaynet.models import build_model                                                           # noqa: E402

R_STRONG = 0.5; N_SITE = 20; F_GRID = np.linspace(0, 0.98, 50); EPOCHS_FIXED = [0, 25, 50, 75, 100, 150, 200, 250, 300]
ARMS = [(f"fcn_mnist_s{s}", a, 300) for s in (42, 43, 44) for a in ("ca", "none")] + [("fcn_mnist_s42", "blur_0.9", 300)] + \
       [(f"cnn_mnist_s{s}", a, 200) for s in (0, 1, 2) for a in ("ca", "none")]


def giant_frac(n, E):
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for a, b in E:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    return np.bincount([find(i) for i in range(n)]).max() / n


def site_fc(n, E, rng):
    """smallest removed fraction at which the mean giant fraction (of the remaining units) falls below half"""
    for f in F_GRID:
        g = []
        for _ in range(N_SITE):
            keep = rng.random(n) >= f; idx = -np.ones(n, int); idx[keep] = np.arange(keep.sum())
            Ek = [(idx[a], idx[b]) for a, b in E if keep[a] and keep[b]]
            g.append(giant_frac(int(keep.sum()), Ek) if keep.sum() > 1 else 0.0)
        if np.mean(g) < 0.5:
            return float(f)
    return float(F_GRID[-1])


def measures(An, nodes, fcn, rng):
    n = An.shape[1]; C = corr(An); iu = np.triu_indices(n, 1); w = C[iu]
    lcc, _ = percolate(C); s = summary(lcc, n); ws = np.sort(w)[::-1]
    r_c = None if s["k_c"] is None else float(ws[int(round(s["k_c"] * n / 2)) - 1])
    sel = w > R_STRONG; E = list(zip(iu[0][sel], iu[1][sel])); k = 2 * len(E) / n
    res = dict(n_live=n, k_c=s["k_c"], r_c=r_c, k_strong=k, ratio=None if s["k_c"] is None else k / s["k_c"],
               giant_strong=float(giant_frac(n, E)), site_fc=site_fc(n, E, rng))
    if fcn:
        pix = nodes % 784; r, c = pix // 28, pix % 28; a, b = iu[0][sel], iu[1][sel]
        res["local_frac"] = float(np.mean(np.maximum(abs(r[a] - r[b]), abs(c[a] - c[b])) <= 2)) if len(a) else None
    return res


def load(run, arm, ep, cache):
    blur = arm.startswith("blur_"); rd = f"{ROOT}/runs/percol_epochs/{run}"
    cfg = load_config(f"{ROOT}/configs/percol_epochs/{run}_{'ca' if blur else arm}.json")
    if run not in cache:
        _, _, te, meta = load_dataset(cfg["data_path"], 1000, cfg["data_name"], val_frac=cfg["val_frac"], split_seed=cfg["split_seed"],
                                      data_norms=cfg["data_norms"], output_path=rd, num_workers=2)
        cache[run] = (torch.cat([x for x, _ in te])[:N_IMG].to(DEV), meta)
    X, meta = cache[run]
    m = (build_blur(cfg, meta, float(arm.split("_")[1])) if blur else build_model(cfg, meta)).to(DEV).eval()
    m.load_state_dict(torch.load(f"{rd}/snapshots/model_{arm}_ep{ep}.pth", map_location=DEV))
    A = hidden(m, X); nodes = np.random.default_rng(0).choice(A.shape[1], N_NODES, replace=False)
    return A[:, torch.from_numpy(nodes).to(DEV)], nodes


def main():
    rng = np.random.default_rng(0); cache = {}; out = dict(r_strong=R_STRONG, arms=[], fixed_set=[])
    for run, arm, ep in ARMS:
        An, nodes = load(run, arm, ep, cache); live = np.nonzero((An.std(0) >= SD_MIN).cpu().numpy())[0]
        res = measures(An[:, torch.from_numpy(live).to(DEV)], nodes[live], run.startswith("fcn"), rng)
        res.update(run=run, arm=arm, epoch=ep); out["arms"].append(res)
        print(f"{run} {arm:8s} ep{ep}: live {res['n_live']}  k_c {res['k_c']:.2f}  r_c {res['r_c']:.3f}  k(r>{R_STRONG}) {res['k_strong']:.1f}  "
              f"ratio {res['ratio']:.2f}  giant {res['giant_strong']:.2f}  site f_c {res['site_fc']:.2f}  local {res.get('local_frac')}", flush=True)
    for run in ("fcn_mnist_s42", "fcn_mnist_s43", "fcn_mnist_s44"):
        for arm in ("none", "ca"):
            An, nodes = load(run, arm, EPOCHS_FIXED[-1], cache); S = np.nonzero((An.std(0) >= SD_MIN).cpu().numpy())[0]
            for ep in EPOCHS_FIXED:
                An, _ = load(run, arm, ep, cache); Sl = S[(An[:, torch.from_numpy(S).to(DEV)].std(0) >= SD_MIN).cpu().numpy()]
                res = measures(An[:, torch.from_numpy(Sl).to(DEV)], nodes[Sl], True, rng)
                res.update(run=run, arm=arm, epoch=ep, n_fixed=len(S)); out["fixed_set"].append(res)
                print(f"FIXED {run} {arm} ep{ep}: {len(Sl)}/{len(S)} live  k_c {res['k_c']}  r_c {res['r_c']}  ratio {res['ratio']}  "
                      f"k(r>{R_STRONG}) {res['k_strong']:.2f}  site f_c {res['site_fc']:.2f}", flush=True)
    json.dump(out, open(f"{ROOT}/diagnostics/percolation_operating_point.json", "w"), indent=1)
    print("WROTE diagnostics/percolation_operating_point.json")


if __name__ == "__main__":
    main()
