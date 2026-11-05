"""Percolation of the co-activation graph, tracked over training, with and without lateral inhibition.

Nodes: a fixed random subsample of N_NODES hidden units (all hidden layers of the FCN; both conv stages,
spatial-channel units, of the CNN). Edges: correlations of the units' activations over a fixed set of
N_IMG test images, added strongest first; after each edge the giant-component fraction is recorded as a
function of the mean degree <k> = 2E/N (Erdos-Renyi percolates at <k> = 1). Two edge weightings: |corr|
(the original analysis) and positive-only (corr > 0; anti-correlations dropped, which removes the
sign confound between the one-sided gated and the symmetric plain activations). Only live units enter the graph:
a unit whose activation s.d. over the image set is below SD_MIN is dropped at that epoch (a near-constant unit's
rescaled rounding-level fluctuations otherwise correlate fully with other such units, and they are not the
network's representation); <k> and the giant-component fraction are over the live units, whose number is recorded.

Per run, arm and epoch: the giant-component fraction on a fixed <k> grid, k_c (the <k> at which the giant component reaches half the nodes), the largest
single-edge jump of the giant-component fraction (staircase amplitude), and the Jaccard overlap of the
edge set at <k> = K_OVERLAP with that of the previous epoch (how much the graph itself changes).
Measurement floor: at the last epoch, k_c and the edge set recomputed on N_FLOOR further disjoint image
sets (the epoch-to-epoch changes are compared with the change caused by the image set alone).

Reads runs/percol_epochs/<run>/snapshots/model_<mech>_ep<E>.pth (scripts/run_snapshots.py, configs in
configs/percol_epochs/) and the run's data_split_indices.json. Writes diagnostics/percolation_epochs.json.
Run: PYTHONPATH=. python paper_code/percolation_epochs.py <run> [<run> ...]
"""
from conwaynet import REPO_ROOT
import glob
import json
import os
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F

from conwaynet.config import load_config
from conwaynet.data import load_dataset
from conwaynet.models import build_model

ROOT = REPO_ROOT; DEV = "cuda"
N_NODES = 1000; N_IMG = 1000; N_FLOOR = 4; K_MAX = 8.0; K_OVERLAP = 3.0; SD_MIN = 1e-3
K_GRID = np.linspace(0.05, K_MAX, 160)                   # <k> grid on which every epoch's curve is stored
OUT = f"{ROOT}/diagnostics/percolation_epochs.json"


@torch.no_grad()
def hidden(m, X):
    """all hidden-layer outputs (after the gate where there is one), flattened and concatenated"""
    if hasattr(m, "linears"):
        x = X.view(X.size(0), -1); acts = []
        for linear, norm in zip(m.linears, m.norms):
            z = torch.tanh(norm(linear(x)))
            if m.gate is not None:
                z = m.gate(x, z)
            acts.append(z); x = z
        return torch.cat(acts, 1)
    z1 = torch.tanh(m.norm1(m.conv1(X)))
    if m.li1 is not None:
        z1 = m.li1(X, z1)
    z2 = F.max_pool2d(torch.tanh(m.norm2(m.conv2(z1))), 2)
    if m.li2 is not None:
        z2 = m.li2(z1, z2)
    return torch.cat([z1.flatten(1), z2.flatten(1)], 1)


def corr(A):
    A = A - A.mean(0, keepdim=True); s = A.std(0, keepdim=True).clamp_min(1e-8); A = A / s
    C = (A.T @ A) / A.shape[0]; C.fill_diagonal_(0.0)
    return C.double().cpu().numpy()


def percolate(W):
    """W symmetric edge weights (higher = added first; <=0 never added). Returns the giant-component
    fraction after each added edge, and the edge set (i<j pairs) at <k> = K_OVERLAP."""
    n = W.shape[0]; iu = np.triu_indices(n, 1); w = W[iu]
    e_max = int(K_MAX * n / 2); order = np.argsort(-w)[:e_max]; order = order[w[order] > 0]
    ei, ej = iu[0][order], iu[1][order]
    parent = np.arange(n); size = np.ones(n, np.int64); giant = 1; lcc = np.empty(len(order))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for t, (a, b) in enumerate(zip(ei, ej)):
        ra, rb = find(a), find(b)
        if ra != rb:
            if size[ra] < size[rb]:
                ra, rb = rb, ra
            parent[rb] = ra; size[ra] += size[rb]; giant = max(giant, size[ra])
        lcc[t] = giant / n
    n_ov = int(K_OVERLAP * n / 2)
    return lcc, set(zip(ei[:n_ov].tolist(), ej[:n_ov].tolist()))


def graph(An):
    """An (images, nodes) activations of the fixed nodes. Percolation over the live nodes (s.d. >= SD_MIN): the
    giant-component fraction per added edge, the edge set at <k> = K_OVERLAP in node indices, the live count."""
    idx = np.nonzero((An.std(0) >= SD_MIN).cpu().numpy())[0]
    lcc, edges = percolate(corr(An[:, torch.from_numpy(idx).to(An.device)]))
    return lcc, {(int(idx[a]), int(idx[b])) for a, b in edges}, len(idx)


def graph_abs(An):
    """as graph(), edges weighted by |correlation|"""
    idx = np.nonzero((An.std(0) >= SD_MIN).cpu().numpy())[0]
    lcc, edges = percolate(np.abs(corr(An[:, torch.from_numpy(idx).to(An.device)])))
    return lcc, {(int(idx[a]), int(idx[b])) for a, b in edges}, len(idx)


def summary(lcc, n):
    """k_c, staircase amplitude, and the giant-component fraction on the fixed <k> grid K_GRID (held at its
    last value if the positive edges run out before K_MAX)"""
    k = 2 * np.arange(1, len(lcc) + 1) / n
    hit = np.nonzero(lcc >= 0.5)[0]
    at = np.minimum(np.round(K_GRID * n / 2).astype(int) - 1, len(lcc) - 1)
    return dict(k_c=float(k[hit[0]]) if len(hit) else None, max_jump=float(np.max(np.diff(lcc, prepend=0.0))),
                lcc_grid=lcc[at].round(4).tolist())


def jaccard(a, b):
    return len(a & b) / len(a | b)


def run(name):
    rd = f"{ROOT}/runs/percol_epochs/{name}"; res = {}
    for mech in ("ca", "none"):
        cfg = load_config(f"{ROOT}/configs/percol_epochs/{name}_{mech}.json")
        _, _, te, meta = load_dataset(cfg["data_path"], 1000, cfg["data_name"], val_frac=cfg["val_frac"], split_seed=cfg["split_seed"],
                                      data_norms=cfg["data_norms"], output_path=rd, num_workers=2)
        X = torch.cat([x for x, _ in te])[: N_IMG * (1 + N_FLOOR)].to(DEV)
        m = build_model(cfg, meta).to(DEV).eval()
        snaps = sorted(glob.glob(f"{rd}/snapshots/model_{mech}_ep*.pth"), key=lambda p: int(re.search(r"ep(\d+)\.pth", p).group(1)))
        epochs = [int(re.search(r"ep(\d+)\.pth", p).group(1)) for p in snaps]
        assert epochs == list(range(cfg["num_epochs"] + 1)), (name, mech, len(epochs))
        nodes = None; rows = {"abs": [], "pos": []}; prev = {"abs": None, "pos": None}
        for ep, p in zip(epochs, snaps):
            m.load_state_dict(torch.load(p, map_location=DEV))
            A = hidden(m, X[:N_IMG])
            if nodes is None:
                nodes = torch.from_numpy(np.random.default_rng(0).choice(A.shape[1], N_NODES, replace=False)).to(DEV)
            An = A[:, nodes]
            for kind, An_k in (("abs", None), ("pos", An)):
                lcc, edges, n_live = graph_abs(An) if kind == "abs" else graph(An_k); s = summary(lcc, n_live)
                s["epoch"] = ep; s["n_live"] = n_live; s["overlap_prev"] = None if prev[kind] is None else jaccard(edges, prev[kind])
                rows[kind].append(s); prev[kind] = edges
            if ep % 25 == 0:
                print(f"{name} {mech} ep{ep}: live {rows['pos'][-1]['n_live']}  k_c pos {rows['pos'][-1]['k_c']}  "
                      f"giant@8 {rows['pos'][-1]['lcc_grid'][-1]}  overlap pos {rows['pos'][-1]['overlap_prev']}", flush=True)
        floor = {"abs": [], "pos": []}; ref = {}
        for j in range(N_FLOOR + 1):                    # last epoch, disjoint image sets (j = 0 is the main set)
            An = hidden(m, X[j * N_IMG:(j + 1) * N_IMG])[:, nodes]
            for kind in ("abs", "pos"):
                lcc, edges, n_live = graph_abs(An) if kind == "abs" else graph(An); s = summary(lcc, n_live)
                if j == 0:
                    ref[kind] = edges
                else:
                    floor[kind].append(dict(k_c=s["k_c"], max_jump=s["max_jump"], giant8=s["lcc_grid"][-1], n_live=n_live, overlap_with_main=jaccard(edges, ref[kind])))
        res[mech] = dict(k_grid=K_GRID.round(4).tolist(), epochs=epochs, abs=rows["abs"], pos=rows["pos"], floor=floor)
    return res


if __name__ == "__main__":
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for name in sys.argv[1:]:
        out[name] = run(name)
        json.dump(out, open(OUT, "w"))
        print(f"WROTE {OUT} [{name}]", flush=True)
