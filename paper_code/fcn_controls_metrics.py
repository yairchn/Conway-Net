"""Appendix A.1-A.2 metrics from the SAVED FCN control checkpoints (paper_code/fcn_controls.py).

Per (arm, seed) in paper_runs/fcn_mnist_controls/<arm>_s<seed>/model.pth: best val acc, held-out
test acc, and on the first 2000 test images
  LFPR       low-frequency power ratio of the class-averaged final hidden maps (radius 0.25)
  lens@1/10  trained classifier applied to hidden layer 1 / 10 (logit lens)
  SSI        scale-separation index (paper_code/scale_separation_index.py; first 1000 test images)
and on the first 1000 test images the co-activation graph (top-100-variance units per layer, |corr|
thresholded to mean degree 6):
  iso        fraction of isolated nodes
  Q          modularity of greedy-modularity communities (the same method for every arm)
  grid_local fraction of edges between units within one Moore step on the 28x28 grid
  coherence / align   active-node map: correlation with its 4-neighbour mean / with the input variance map
Plus the model-free FiLM-gain numbers of the plain-blend gate at layer 1 (gain = S(m) - B(m)).
Writes paper_runs/fcn_mnist_controls/_metrics.json and prints mean +- std per arm.
Run: PYTHONPATH=. python paper_code/fcn_controls_metrics.py
"""
from conwaynet import REPO_ROOT
import importlib.util
import json
import os

import networkx as nx
import numpy as np
import torch

import sys
sys.path.insert(0, f"{REPO_ROOT}/paper_code")
import scale_separation_index as SS
from conwaynet.config import load_config
from conwaynet.data import load_dataset
from conwaynet.lateral_inhibition import _rule_mask, moore_mean, parse_rule

ROOT = REPO_ROOT; DEV = "cuda"; HW = 28
spec = importlib.util.spec_from_file_location("fcn_controls", f"{ROOT}/paper_code/fcn_controls.py")
fc = importlib.util.module_from_spec(spec); spec.loader.exec_module(fc)
EDGES = [0, 2, 4, 6, 9, 13, 20]; NB = 6; UNITS, TD = 100, 6.0


@torch.no_grad()
def hidden(m, X):
    cur = X.view(X.size(0), -1); outs = []
    for i, lin in enumerate(m.linears):
        t = lin(cur); t = m.norms[i](t) if m.norms is not None else t; z = torch.tanh(t)
        if m.gate is not None:
            z = m.gate(cur, z)
        cur = z; outs.append(cur)
    return outs


def lfpr(maps, rf=0.25):
    x = maps - maps.mean((-2, -1), keepdims=True)
    P = np.abs(np.fft.fftshift(np.fft.fft2(x, axes=(-2, -1)), axes=(-2, -1))) ** 2
    yy, xx = np.meshgrid(np.arange(HW), np.arange(HW), indexing="ij")
    r = np.sqrt((yy - (HW - 1) / 2) ** 2 + (xx - (HW - 1) / 2) ** 2); r = r / r.max()
    return (P * (r <= rf)).sum((-2, -1)) / (P.sum((-2, -1)) + 1e-8)


def ssi(m, X):
    """Scale-separation index, the shared definition (paper_code/scale_separation_index.py)."""
    return SS.scale_separation(torch.stack([c.view(c.size(0), HW, HW) for c in hidden(m, X[:SS.N_IMAGES])]))["ssi"]


def graph_metrics(A, dv):
    cols, uid = [], []
    for a in A:
        idx = np.argsort(a.var(0))[-min(UNITS, a.shape[1]):]; cols.append(a[:, idx]); uid += list(idx)
    Xc = np.concatenate(cols, 1); C = np.nan_to_num(np.corrcoef(Xc.T)); np.fill_diagonal(C, 0.0)
    n = C.shape[0]; iu = np.triu_indices(n, 1); mg = np.abs(C[iu]); thr = np.sort(mg)[-int(TD * n / 2)]
    Gr = nx.Graph(); Gr.add_nodes_from(range(n)); sel = mg >= thr
    Gr.add_edges_from(zip(iu[0][sel].tolist(), iu[1][sel].tolist()))
    uid = np.array(uid); r, c = np.divmod(uid, HW)
    iso = sum(1 for v in Gr if Gr.degree(v) == 0) / n
    Q = nx.community.modularity(Gr, nx.community.greedy_modularity_communities(Gr))
    e = np.array(Gr.edges())
    loc = float(np.mean(np.maximum(abs(r[e[:, 0]] - r[e[:, 1]]), abs(c[e[:, 0]] - c[e[:, 1]])) <= 1)) if len(e) else 0.0
    mp = np.zeros((HW, HW))
    for v in Gr:
        if Gr.degree(v) > 0: mp[r[v], c[v]] += 1
    nb = (np.roll(mp, 1, 0) + np.roll(mp, -1, 0) + np.roll(mp, 1, 1) + np.roll(mp, -1, 1)) / 4
    corr = lambda a, b: float(((a - a.mean()) * (b - b.mean())).sum() / np.sqrt(((a - a.mean()) ** 2).sum() * ((b - b.mean()) ** 2).sum() + 1e-12))
    return dict(iso=iso, Q=float(Q), grid_local=loc, coherence=corr(mp.ravel(), nb.ravel()), align=corr(mp.ravel(), dv.ravel()))


cfg = load_config(fc.CFG.format("ca"))
_, _, te, meta = load_dataset(cfg["data_path"], 500, cfg["data_name"], val_frac=cfg["val_frac"], split_seed=cfg["split_seed"],
                              data_norms=cfg["data_norms"], output_path=None, num_workers=2)
X, Y = [], []
for xb, yb in te:
    X.append(xb); Y.append(yb)
    if sum(len(x) for x in X) >= 2000: break
X, Y = torch.cat(X)[:2000].to(DEV), torch.cat(Y)[:2000].to(DEV)
dv = X[:1000].view(-1, HW, HW).var(0).cpu().numpy()

res = {}
for d in sorted(os.listdir(fc.OUT)):
    p = f"{fc.OUT}/{d}/model.pth"
    if not os.path.exists(p): continue
    arm, seed = d.rsplit("_s", 1)
    m, _ = fc.build(arm, int(seed), meta)
    m.load_state_dict(torch.load(p, map_location="cpu")); m.to(DEV).eval()
    r = json.load(open(f"{fc.OUT}/{d}/results.json"))
    H = hidden(m, X)
    final = H[-1].cpu().numpy(); yv = Y.cpu().numpy()
    with torch.no_grad():
        lens = [float((m.classifier(h).argmax(1) == Y).float().mean()) * 100 for h in H]
    row = dict(arm=arm, seed=int(seed), val=r["best_val_acc"], test=r["test_clean_acc"],
               LFPR=float(lfpr(np.stack([final[yv == k].mean(0).reshape(HW, HW) for k in range(10)])).mean()),
               lens1=lens[0], lens10=lens[-1], SSI=float(ssi(m, X)))
    row.update(graph_metrics([h[:1000].cpu().numpy() for h in H], dv))
    res[d] = row
    print(d, {k: round(v, 3) for k, v in row.items() if isinstance(v, float)}, flush=True)
    del m; torch.cuda.empty_cache()

# model-free FiLM gain of the plain-blend gate at layer 1: y = gamma z + beta, gamma = S(m) - B(m)
b, s = parse_rule(cfg["rule"])
occ = (X.view(-1, 1, HW, HW) + 1) / 2
mm = moore_mean(occ)
gain = (_rule_mask(mm, s, cfg["delta"], cfg["gamma"]) - _rule_mask(mm, b, cfg["delta"], cfg["gamma"])).abs()
stroke = occ > 0.1
film = dict(mean_abs_gain=float(gain.mean()), on_stroke=float(gain[stroke].mean()), off_stroke=float(gain[~stroke].mean()),
            frac_nonzero=float((gain > 0.01).float().mean()))
print("FiLM gain (layer 1, model-free):", {k: round(v, 3) for k, v in film.items()})

agg = {}
for row in res.values():
    agg.setdefault(row["arm"], []).append(row)
print(f"\n{'arm':12s} n " + " ".join(f"{k:>12s}" for k in ("val", "test", "LFPR", "lens1", "lens10", "SSI", "iso", "Q", "grid_local", "coherence", "align")))
summary = {}
for arm, rows in agg.items():
    summary[arm] = {k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows])), len(rows))
                    for k in ("val", "test", "LFPR", "lens1", "lens10", "SSI", "iso", "Q", "grid_local", "coherence", "align")}
    print(f"{arm:12s} {len(rows)} " + " ".join(f"{summary[arm][k][0]:6.3f}±{summary[arm][k][1]:.3f}" for k in summary[arm]))
json.dump(dict(per_run=res, summary=summary, film_gain=film), open(f"{fc.OUT}/_metrics.json", "w"), indent=2)
print("WROTE", f"{fc.OUT}/_metrics.json")
