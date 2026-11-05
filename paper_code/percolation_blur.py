"""Percolation over training for the blur control (paper_code/percolation_blur_snapshots.py), measured exactly as
paper_code/percolation_epochs.py measures the with- and without-LI networks (same nodes, images, positive edges,
readouts, measurement floor), plus the layer composition of the giant component at <k> = K_FIX as in
paper_code/percolation_giant_layers.py (top-layer share, same-layer edge fraction), at every epoch.
Writes diagnostics/percolation_blur.json, then prints the plateau summary next to the with- and without-LI arms
of diagnostics/percolation_epochs.json.
Run: PYTHONPATH=. python paper_code/percolation_blur.py <run> <lam> [<lam> ...]
"""
from conwaynet import REPO_ROOT
import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, f"{REPO_ROOT}/paper_code")
from percolation_epochs import hidden, corr, percolate, summary, graph, SD_MIN, jaccard, N_NODES, N_IMG, N_FLOOR, K_GRID, ROOT, DEV  # noqa: E402
from percolation_blur_snapshots import build_blur                # noqa: E402
from conwaynet.config import load_config                         # noqa: E402
from conwaynet.data import load_dataset                          # noqa: E402
from conwaynet.models import build_model                         # noqa: E402

OUT = f"{ROOT}/diagnostics/percolation_blur.json"; K_FIX = 4.0


def composition(C, layer, n_layers):
    n = C.shape[0]; iu = np.triu_indices(n, 1); w = C[iu]
    o = np.argsort(-w)[: int(K_FIX * n / 2)]; o = o[w[o] > 0]; ei, ej = iu[0][o], iu[1][o]
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for a, b in zip(ei, ej):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    roots = np.array([find(i) for i in range(n)]); ig = roots == np.bincount(roots).argmax()
    p = np.bincount(layer[ig], minlength=n_layers) / ig.sum()
    return dict(giant=float(ig.mean()), top_share=float(p.max()), top_layer=int(p.argmax()), same_layer=float(np.mean(layer[ei] == layer[ej])))


def run(name, arm, cfg, build):
    rd = f"{ROOT}/runs/percol_epochs/{name}"
    _, _, te, meta = load_dataset(cfg["data_path"], 1000, cfg["data_name"], val_frac=cfg["val_frac"], split_seed=cfg["split_seed"],
                                  data_norms=cfg["data_norms"], output_path=rd, num_workers=2)
    X = torch.cat([x for x, _ in te])[: N_IMG * (1 + N_FLOOR)].to(DEV)
    m = build(cfg, meta).to(DEV).eval(); epochs = list(range(cfg["num_epochs"] + 1))
    for ep in epochs:
        assert os.path.exists(f"{rd}/snapshots/model_{arm}_ep{ep}.pth"), (name, arm, ep)
    nodes = None; rows = []; comp = []; prev = None
    for ep in epochs:
        m.load_state_dict(torch.load(f"{rd}/snapshots/model_{arm}_ep{ep}.pth", map_location=DEV)); A = hidden(m, X[:N_IMG])
        if nodes is None:
            nodes = np.random.default_rng(0).choice(A.shape[1], N_NODES, replace=False)
            if hasattr(m, "linears"):
                layer, n_layers = nodes // m.linears[0].out_features, len(m.linears)
            else:
                with torch.no_grad():
                    n1 = m.conv1(X[:1]).numel()
                layer, n_layers = (nodes >= n1).astype(int), 2
            tn = torch.from_numpy(nodes).to(DEV)
        An = A[:, tn]; lcc, edges, n_live = graph(An); s = summary(lcc, n_live)
        s["epoch"] = ep; s["n_live"] = n_live; s["overlap_prev"] = None if prev is None else jaccard(edges, prev); rows.append(s); prev = edges
        live = np.nonzero((An.std(0) >= SD_MIN).cpu().numpy())[0]
        comp.append(dict(epoch=ep, n_live=n_live, **composition(corr(An[:, torch.from_numpy(live).to(DEV)]), layer[live], n_layers)))
        if ep % 25 == 0:
            print(f"{name} {arm} ep{ep}: live {n_live}  k_c {s['k_c']}  giant@8 {s['lcc_grid'][-1]:.2f}  top-layer share {comp[-1]['top_share']:.2f}", flush=True)
    floor = []; ref = None
    for j in range(N_FLOOR + 1):
        lcc, edges, n_live = graph(hidden(m, X[j * N_IMG:(j + 1) * N_IMG])[:, tn]); s = summary(lcc, n_live)
        if j == 0:
            ref = edges
        else:
            floor.append(dict(k_c=s["k_c"], max_jump=s["max_jump"], giant8=s["lcc_grid"][-1], n_live=n_live, overlap_with_main=jaccard(edges, ref)))
    return dict(k_grid=K_GRID.round(4).tolist(), epochs=epochs, pos=rows, floor=floor, composition=comp)


def plateau(rows, floor, E):
    P = [x for x in rows if x["epoch"] >= E // 4]
    kc = np.array([np.nan if x["k_c"] is None else x["k_c"] for x in P], dtype=float)
    return (f"live {np.mean([x['n_live'] for x in P]):.0f}  k_c {np.nanmean(kc):.2f}  giant@8 {np.mean([x['lcc_grid'][-1] for x in P]):.2f}  overlap {np.mean([x['overlap_prev'] for x in P]):.3f}  "
            f"jump {np.mean([x['max_jump'] for x in P]):.3f}  floor overlap {np.mean([x['overlap_with_main'] for x in floor]):.3f}")


if __name__ == "__main__":
    name = sys.argv[1]; out = json.load(open(OUT)) if os.path.exists(OUT) else {}; out.setdefault(name, {})
    cfg_ca = load_config(f"{ROOT}/configs/percol_epochs/{name}_ca.json")
    arms = [(f"blur_{lam:g}", cfg_ca, lambda c, mt, lam=lam: build_blur(c, mt, lam)) for lam in map(float, sys.argv[2:])]
    arms += [(mech, load_config(f"{ROOT}/configs/percol_epochs/{name}_{mech}.json"), build_model) for mech in ("ca", "none")]
    for arm, cfg, build in arms:              # with- and without-LI rerun here for the layer composition on the same footing
        out[name][arm] = run(name, arm, cfg, build)
        json.dump(out, open(OUT, "w")); print("WROTE", OUT, flush=True)
    print(f"\n{name}, positive edges, plateau = epochs >= {cfg_ca['num_epochs'] // 4}")
    for arm, r in out[name].items():
        P = [x for x in r["composition"] if x["epoch"] >= r["epochs"][-1] // 4]
        print(f"  {arm:9s} {plateau(r['pos'], r['floor'], r['epochs'][-1])}  top-layer share {np.mean([x['top_share'] for x in P]):.2f}  "
              f"same-layer edges {np.mean([x['same_layer'] for x in P]):.2f}")
