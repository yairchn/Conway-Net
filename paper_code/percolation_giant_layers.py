"""Layer composition of the giant component of the FCN co-activation graph over training (positive-correlation
edges, the strongest K_FIX*N/2 kept; same nodes, images and snapshots as paper_code/percolation_epochs.py).
Per run, arm and epoch: giant-component fraction; share of the giant in its most common layer; number of layers
holding >= 5% of the giant; normalised layer entropy of the giant (1 = spread evenly over the ten layers); and,
for all kept edges, the fraction joining two units of the same layer (random pairs: 0.1).
Run: PYTHONPATH=. python paper_code/percolation_giant_layers.py <k_fix> <run> [<run> ...]
"""
from conwaynet import REPO_ROOT
import sys

import numpy as np
import torch

sys.path.insert(0, f"{REPO_ROOT}/paper_code")
from percolation_epochs import hidden, corr, N_NODES, N_IMG, ROOT, DEV  # noqa: E402
from conwaynet.config import load_config  # noqa: E402
from conwaynet.data import load_dataset  # noqa: E402
from conwaynet.models import build_model  # noqa: E402

K_FIX = float(sys.argv[1]); EPOCHS = [0, 10, 20, 25, 30, 35, 50, 100, 200, 300]


def giant(n, E):
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for a, b in E:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    roots = np.array([find(i) for i in range(n)])
    return roots == np.bincount(roots).argmax()


for run in sys.argv[2:]:
    rd = f"{ROOT}/runs/percol_epochs/{run}"
    for mech in ("none", "ca"):
        cfg = load_config(f"{ROOT}/configs/percol_epochs/{run}_{mech}.json")
        _, _, te, meta = load_dataset(cfg["data_path"], 1000, cfg["data_name"], val_frac=cfg["val_frac"], split_seed=cfg["split_seed"],
                                      data_norms=cfg["data_norms"], output_path=rd, num_workers=2)
        X = torch.cat([x for x, _ in te])[:N_IMG].to(DEV); m = build_model(cfg, meta).to(DEV).eval()
        n = N_NODES; iu = np.triu_indices(n, 1); nodes = None
        print(f"\n{run} {mech}  (<k> = {K_FIX:g})\n  epoch  giant  top-layer share  layers>=5%  layer entropy  same-layer edges")
        for ep in EPOCHS:
            m.load_state_dict(torch.load(f"{rd}/snapshots/model_{mech}_ep{ep}.pth", map_location=DEV)); A = hidden(m, X)
            if nodes is None:
                nodes = np.random.default_rng(0).choice(A.shape[1], n, replace=False); layer = nodes // m.linears[0].out_features
                L = len(m.linears)
            C = corr(A[:, torch.from_numpy(nodes).to(DEV)]); w = C[iu]
            o = np.argsort(-w)[: int(K_FIX * n / 2)]; o = o[w[o] > 0]; E = np.stack([iu[0][o], iu[1][o]], 1)
            ig = giant(n, E); cnt = np.bincount(layer[ig], minlength=L); p = cnt / cnt.sum()
            ent = -(p[p > 0] * np.log(p[p > 0])).sum() / np.log(L)
            same = np.mean(layer[E[:, 0]] == layer[E[:, 1]])
            print(f"  {ep:5d}  {ig.mean():.2f}   {p.max():.2f} (layer {p.argmax() + 1:2d})    {int((p >= 0.05).sum()):2d}         {ent:.2f}          {same:.2f}")
