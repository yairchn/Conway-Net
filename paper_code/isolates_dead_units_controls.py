"""The co-activation-graph metrics of Appendix "grid rule" (paper_code/fcn_controls_metrics.py: top-UNITS-variance
units per layer, |corr| edges at mean degree TD, first 1000 test images) on every arm of the saved FCN controls,
counting all nodes (as in the appendix) and active nodes only (activation s.d. over the images >= SD_MIN; graph
rebuilt on them at the same mean degree). Per arm and seed: fraction of nodes that are inactive, isolate fraction,
modularity Q (greedy modularity) and grid-local edge fraction, both ways.
Writes diagnostics/isolates_dead_units_controls.json.
Run: PYTHONPATH=. python paper_code/isolates_dead_units_controls.py
"""
from conwaynet import REPO_ROOT
import importlib.util
import json
import os

import networkx as nx
import numpy as np
import torch

from conwaynet.config import load_config
from conwaynet.data import load_dataset

ROOT = REPO_ROOT; DEV = "cuda"; HW = 28; UNITS, TD = 100, 6.0; SD_MIN = 1e-3; N = 1000
ARMS = ["ca", "none", "localmix", "randnb", "shuffled_m", "blur_0.5", "blur_0.9"]
spec = importlib.util.spec_from_file_location("fcn_controls", f"{ROOT}/paper_code/fcn_controls.py")
fc = importlib.util.module_from_spec(spec); spec.loader.exec_module(fc)


@torch.no_grad()
def hidden(m, X):
    cur = X.view(X.size(0), -1); outs = []
    for i, lin in enumerate(m.linears):
        t = lin(cur); t = m.norms[i](t) if m.norms is not None else t; z = torch.tanh(t)
        if m.gate is not None:
            z = m.gate(cur, z)
        cur = z; outs.append(cur.cpu().numpy())
    return outs


def metrics(Xc, uid):
    C = np.nan_to_num(np.corrcoef(Xc.T)); np.fill_diagonal(C, 0.0)
    n = C.shape[0]; iu = np.triu_indices(n, 1); mg = np.abs(C[iu]); thr = np.sort(mg)[-int(TD * n / 2)]
    sel = mg >= thr; Gr = nx.Graph(); Gr.add_nodes_from(range(n)); Gr.add_edges_from(zip(iu[0][sel].tolist(), iu[1][sel].tolist()))
    r, c = np.divmod(uid, HW); e = np.array(Gr.edges())
    return dict(iso=sum(1 for v in Gr if Gr.degree(v) == 0) / n,
                Q=float(nx.community.modularity(Gr, nx.community.greedy_modularity_communities(Gr))),
                grid_local=float(np.mean(np.maximum(abs(r[e[:, 0]] - r[e[:, 1]]), abs(c[e[:, 0]] - c[e[:, 1]])) <= 1)))


def main():
    cfg = load_config(fc.CFG.format("ca"))
    _, _, te, meta = load_dataset(cfg["data_path"], 500, cfg["data_name"], val_frac=cfg["val_frac"], split_seed=cfg["split_seed"],
                                  data_norms=cfg["data_norms"], output_path=None, num_workers=2)
    X = torch.cat([x for x, _ in te])[:N].to(DEV); res = {}
    for arm in ARMS:
        for seed in (0, 1, 2):
            p = f"{fc.OUT}/{arm}_s{seed}/model.pth"
            m, _ = fc.build(arm, seed, meta); m.load_state_dict(torch.load(p, map_location="cpu")); m.to(DEV).eval()
            cols, uid = [], []
            for a in hidden(m, X):
                idx = np.argsort(a.var(0))[-UNITS:]; cols.append(a[:, idx]); uid += list(idx)
            Xc = np.concatenate(cols, 1); uid = np.array(uid); live = Xc.std(0) >= SD_MIN
            row = dict(inactive=float(1 - live.mean()), all=metrics(Xc, uid), active=metrics(Xc[:, live], uid[live]))
            res[f"{arm}_s{seed}"] = row
            print(f"{arm:10s} s{seed}: inactive {row['inactive']:.2f} | all: iso {row['all']['iso']:.2f} Q {row['all']['Q']:.2f} "
                  f"grid-local {row['all']['grid_local']:.2f} | active: iso {row['active']['iso']:.2f} Q {row['active']['Q']:.2f} "
                  f"grid-local {row['active']['grid_local']:.2f}", flush=True)
            del m; torch.cuda.empty_cache()
    json.dump(res, open(f"{ROOT}/diagnostics/isolates_dead_units_controls.json", "w"), indent=1)


if __name__ == "__main__":
    main()
