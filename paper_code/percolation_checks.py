"""Checks on the percolation-over-training analysis (paper_code/percolation_epochs.py, percolation_blur.py) raised in
review. Same nodes (default_rng(seed).choice), images (first test images, unshuffled loader) and positive edges.
  forward    hidden() reproduces the model's forward pass (classifier(last hidden layer) == m(X)).
  collapse   plain FCN: for the layer hosting the giant component at <k> = 8, the fraction of its giant units'
             activations at the tanh rail (|z| > 0.99), their s.d., mean pairwise correlation inside the giant, the
             layer's effective rank; and the giant / top-layer share when (a) Spearman (rank) correlation replaces
             Pearson, (b) units with s.d. < 1e-3 are dropped, (c) units correlated above 0.99 are merged into one
             node, (d) edges are taken above a fixed correlation threshold instead of a fixed budget; plus the
             99.9th percentile of within- and cross-layer correlations. Arms: plain, with-LI and blur FCN.
  spatial    FCN: fraction of the strongest edges at <k> = 3 joining units at the same pixel (any layers), within
             Chebyshev pixel distance 2, and in the same layer, against random pairs; and a spatial reference
             graph on the same nodes (edges added by increasing pixel distance, random tie-break) next to Erdos-Renyi.
  subsample  plain and with-LI FCN with node subsamples seeds 1 and 2: collapse epoch, host layer, plateau k_c.
  floor      every readout (k_c, largest jump, giant at <k> = 8, edge overlap) across four disjoint image sets at
             the last epoch, all runs and arms, next to the plateau s.d. over epochs.
Writes diagnostics/percolation_checks.json. Run: PYTHONPATH=. python paper_code/percolation_checks.py
"""
from conwaynet import REPO_ROOT
import json
import sys

import numpy as np
import torch

sys.path.insert(0, f"{REPO_ROOT}/paper_code")
from percolation_epochs import hidden, corr, percolate, summary, jaccard, N_NODES, N_IMG, N_FLOOR, ROOT, DEV  # noqa: E402
from percolation_blur_snapshots import build_blur                # noqa: E402
from conwaynet.config import load_config                         # noqa: E402
from conwaynet.data import load_dataset                          # noqa: E402
from conwaynet.models import build_model                         # noqa: E402

HW = 28; RES = {}


def arm_model(run, arm):
    if arm.startswith("blur_"):
        cfg = load_config(f"{ROOT}/configs/percol_epochs/{run}_ca.json"); return cfg, lambda meta: build_blur(cfg, meta, float(arm.split("_")[1]))
    cfg = load_config(f"{ROOT}/configs/percol_epochs/{run}_{arm}.json"); return cfg, lambda meta: build_model(cfg, meta)


def setup(run, arm):
    cfg, build = arm_model(run, arm)
    _, _, te, meta = load_dataset(cfg["data_path"], 1000, cfg["data_name"], val_frac=cfg["val_frac"], split_seed=cfg["split_seed"],
                                  data_norms=cfg["data_norms"], output_path=f"{ROOT}/runs/percol_epochs/{run}", num_workers=2)
    X = torch.cat([x for x, _ in te])[: N_IMG * (1 + N_FLOOR)].to(DEV)
    return cfg, build(meta).to(DEV).eval(), X


def load(m, run, arm, ep):
    m.load_state_dict(torch.load(f"{ROOT}/runs/percol_epochs/{run}/snapshots/model_{arm}_ep{ep}.pth", map_location=DEV)); return m


def giant_mask(n, ei, ej):
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for a, b in zip(ei, ej):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    roots = np.array([find(i) for i in range(n)]); return roots == np.bincount(roots).argmax()


def budget_edges(C, k):
    n = C.shape[0]; iu = np.triu_indices(n, 1); w = C[iu]; o = np.argsort(-w)[: int(k * n / 2)]; o = o[w[o] > 0]
    return iu[0][o], iu[1][o]


def giant_info(C, layer, k=8.0):
    ei, ej = budget_edges(C, k); g = giant_mask(C.shape[0], ei, ej); p = np.bincount(layer[g], minlength=10) / g.sum()
    return g, dict(giant=float(g.mean()), top_share=float(p.max()), host=int(p.argmax()))


def eff_rank(A):
    s = torch.linalg.svdvals(A - A.mean(0, keepdim=True)).double().cpu().numpy(); p = s / s.sum()
    return float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()))


def rank_transform(A):
    return A.argsort(0).argsort(0).float()


# ------------------------------------------------------------------------------------------- forward
RES["forward"] = {}
for run in ("fcn_mnist_s42", "cnn_mnist_s0"):
    for arm in ("ca", "none", "blur_0.9"):
        cfg, m, X = setup(run, arm); m = load(m, run, arm, cfg["num_epochs"])
        with torch.no_grad():
            H = hidden(m, X[:256]); last = H[:, -m.classifier.in_features:]
            err = float((m.classifier(last) - m(X[:256])).abs().max())
        RES["forward"][f"{run}/{arm}"] = err; print(f"forward check {run} {arm}: max |diff| {err:.2e}", flush=True)
        assert err < 1e-4

# ------------------------------------------------------------------------------------------- collapse
EPOCHS = [0, 20, 25, 30, 35, 50, 100, 200, 300]; RES["collapse"] = {}
print("\nCOLLAPSE (FCN s42, <k>=8 unless stated)")
for arm in ("none", "ca", "blur_0.9"):
    cfg, m, X = setup("fcn_mnist_s42", arm); Xi = X[:N_IMG]; rows = []
    print(f"\n {arm}\n  ep | giant top(host) | rail  sd   corr-in-giant | effrank host | spearman g/top | drop-sd g/top | merge>.99 n,g/top | thr.5 g,E thr.8 g,E | q99.9 in/cross")
    for ep in EPOCHS:
        load(m, "fcn_mnist_s42", arm, ep); A = hidden(m, Xi); W = m.linears[0].out_features
        nodes = np.random.default_rng(0).choice(A.shape[1], N_NODES, replace=False); layer = nodes // W
        An = A[:, torch.from_numpy(nodes).to(DEV)]; C = corr(An); g, gi = giant_info(C, layer)
        Ag = An[:, torch.from_numpy(np.nonzero(g)[0]).to(DEV)]
        rail = float((Ag.abs() > 0.99).float().mean()); sd = float(Ag.std(0).median())
        Cg = C[np.ix_(g, g)]; cin = float(Cg[np.triu_indices(g.sum(), 1)].mean()) if g.sum() > 1 else np.nan
        hl = gi["host"]; er = eff_rank(A[:, hl * W:(hl + 1) * W])
        _, sp = giant_info(corr(rank_transform(An)), layer)
        sdv = An.std(0).cpu().numpy(); keep = sdv >= 1e-3; Ck = C[np.ix_(keep, keep)]; _, dr = giant_info(Ck, layer[keep])
        # merge near-duplicates: connected components of the corr > 0.99 graph, one representative each
        iu = np.triu_indices(N_NODES, 1); hi = C[iu] > 0.99; dm = giant_mask(N_NODES, iu[0][hi], iu[1][hi])
        parent = np.arange(N_NODES)

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]; a = parent[a]
            return a
        for a, b in zip(iu[0][hi], iu[1][hi]):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
        reps = np.unique([find(i) for i in range(N_NODES)]); Cm = C[np.ix_(reps, reps)]
        eim, ejm = budget_edges(Cm, 8.0); gm = giant_mask(len(reps), eim, ejm)
        pm = np.bincount(layer[reps][gm], minlength=10) / gm.sum()
        thr = {}
        for t in (0.5, 0.8):
            sel = C[iu] > t; gt = giant_mask(N_NODES, iu[0][sel], iu[1][sel]); thr[t] = (float(gt.mean()), int(sel.sum()))
        same = layer[iu[0]] == layer[iu[1]]; w = C[iu]
        q_in, q_x = float(np.quantile(w[same], 0.999)), float(np.quantile(w[~same], 0.999))
        r = dict(epoch=ep, **gi, rail=rail, sd_median=sd, corr_in_giant=cin, eff_rank_host=er, spearman=sp, drop_low_sd=dict(n_dropped=int((~keep).sum()), **dr),
                 merged=dict(n_nodes=int(len(reps)), giant=float(gm.mean()), top_share=float(pm.max())),
                 threshold={str(t): dict(giant=v[0], n_edges=v[1]) for t, v in thr.items()}, q999_within=q_in, q999_cross=q_x)
        rows.append(r)
        print(f"  {ep:3d} | {gi['giant']:.2f} {gi['top_share']:.2f}(L{hl+1}) | {rail:.2f} {sd:.3f} {cin:.2f} | {er:6.1f} | {sp['giant']:.2f}/{sp['top_share']:.2f} | "
              f"{dr['giant']:.2f}/{dr['top_share']:.2f} ({int((~keep).sum())} drop) | {len(reps)},{gm.mean():.2f}/{pm.max():.2f} | "
              f"{thr[0.5][0]:.2f},{thr[0.5][1]} {thr[0.8][0]:.2f},{thr[0.8][1]} | {q_in:.2f}/{q_x:.2f}", flush=True)
    RES["collapse"][arm] = rows

# ------------------------------------------------------------------------------------------- spatial
print("\nSPATIAL (FCN s42, strongest edges at <k>=3)")
nodes = np.random.default_rng(0).choice(10 * HW * HW, N_NODES, replace=False); layer = nodes // (HW * HW); pix = nodes % (HW * HW)
r_, c_ = pix // HW, pix % HW; iu = np.triu_indices(N_NODES, 1)
cheb = np.maximum(np.abs(r_[iu[0]] - r_[iu[1]]), np.abs(c_[iu[0]] - c_[iu[1]]))
base = dict(same_pixel=float(np.mean(cheb == 0)), within2=float(np.mean(cheb <= 2)), same_layer=float(np.mean(layer[iu[0]] == layer[iu[1]])))
print(f"  random pairs: same pixel {base['same_pixel']:.3f}  within 2px {base['within2']:.3f}  same layer {base['same_layer']:.3f}")
RES["spatial"] = {"random_pairs": base}
lin = {(a, b): i for i, (a, b) in enumerate(zip(iu[0], iu[1]))}
for arm in ("ca", "blur_0.9", "blur_0.5", "none"):
    cfg, m, X = setup("fcn_mnist_s42", arm); out = {}
    for ep in (100, cfg["num_epochs"]):
        load(m, "fcn_mnist_s42", arm, ep); C = corr(hidden(m, X[:N_IMG])[:, torch.from_numpy(nodes).to(DEV)])
        ei, ej = budget_edges(C, 3.0); d = np.maximum(np.abs(r_[ei] - r_[ej]), np.abs(c_[ei] - c_[ej]))
        out[ep] = dict(same_pixel=float(np.mean(d == 0)), within2=float(np.mean(d <= 2)), same_layer=float(np.mean(layer[ei] == layer[ej])),
                       same_pixel_cross_layer=float(np.mean((d == 0) & (layer[ei] != layer[ej]))))
        print(f"  {arm:9s} ep{ep}: same pixel {out[ep]['same_pixel']:.3f} (cross-layer {out[ep]['same_pixel_cross_layer']:.3f})  within 2px {out[ep]['within2']:.3f}  same layer {out[ep]['same_layer']:.3f}", flush=True)
    RES["spatial"][arm] = out
# spatial reference: same nodes, edges by increasing pixel distance (Euclidean, random tie-break), and 2-D RGG
rng = np.random.default_rng(0); eu = np.hypot(r_[iu[0]] - r_[iu[1]], c_[iu[0]] - c_[iu[1]])
refs = {}
for name, dist in (("pixel_distance_same_nodes", eu + 1e-6 * rng.random(len(eu))),):
    W = np.zeros((N_NODES, N_NODES)); W[iu] = -dist; W = W + W.T
    W[iu] -= W[iu].min() - 1; W.T[iu] = W[iu]
    lcc, _ = percolate(W); refs[name] = summary(lcc, N_NODES)
P = rng.random((N_NODES, 2)); D = np.hypot(*(P[iu[0]] - P[iu[1]]).T); W = np.zeros((N_NODES, N_NODES)); W[iu] = 2 - D; W = W + W.T
lcc, _ = percolate(W); refs["rgg_2d"] = summary(lcc, N_NODES)
P = rng.random((N_NODES, 3)); D = np.linalg.norm(P[iu[0]] - P[iu[1]], axis=1); W = np.zeros((N_NODES, N_NODES)); W[iu] = 2 - D; W = W + W.T
lcc, _ = percolate(W); refs["rgg_3d"] = summary(lcc, N_NODES)
for k, v in refs.items():
    print(f"  reference {k}: k_c {v['k_c']}  largest jump {v['max_jump']:.3f}  giant@8 {v['lcc_grid'][-1]:.2f}")
RES["spatial"]["references"] = refs

# ------------------------------------------------------------------------------------------- subsample
print("\nSUBSAMPLE (FCN s42, node seeds 0/1/2)")
RES["subsample"] = {}
for arm in ("none", "ca"):
    cfg, m, X = setup("fcn_mnist_s42", arm); res = {s: [] for s in (0, 1, 2)}
    for ep in range(0, cfg["num_epochs"] + 1, 5):
        load(m, "fcn_mnist_s42", arm, ep); A = hidden(m, X[:N_IMG])
        for s in (0, 1, 2):
            nodes_s = np.random.default_rng(s).choice(A.shape[1], N_NODES, replace=False); C = corr(A[:, torch.from_numpy(nodes_s).to(DEV)])
            _, gi = giant_info(C, nodes_s // m.linears[0].out_features); lcc, _ = percolate(C)
            res[s].append(dict(epoch=ep, k_c=summary(lcc, N_NODES)["k_c"], **gi))
    for s, rows in res.items():
        coll = next((r["epoch"] for r in rows if r["epoch"] > 0 and r["giant"] < 0.5), None)
        hosts = [f"{r['epoch']}:L{r['host']+1}" for r in rows if r["epoch"] in (35, 50, 100, 200, 300)]
        kc = [r["k_c"] for r in rows if r["epoch"] >= 75 and r["k_c"] is not None]
        print(f"  {arm:4s} node seed {s}: first epoch giant@8<0.5: {coll}  host layers {' '.join(hosts)}  plateau k_c {np.mean(kc) if kc else float('nan'):.2f}", flush=True)
    RES["subsample"][arm] = {str(s): v for s, v in res.items()}

# ------------------------------------------------------------------------------------------- floor
print("\nFLOOR (last epoch, 4 disjoint image sets; plateau s.d. over epochs from the per-epoch JSONs)")
PE = json.load(open(f"{ROOT}/diagnostics/percolation_epochs.json")); PB = json.load(open(f"{ROOT}/diagnostics/percolation_blur.json"))
RES["floor"] = {}
for run in PE:
    arms = ["ca", "none"] + ([a for a in PB[run] if a.startswith("blur_")] if run in PB else [])
    for arm in arms:
        cfg, m, X = setup(run, arm); load(m, run, arm, cfg["num_epochs"]); A0 = hidden(m, X[:N_IMG])
        nodes_s = torch.from_numpy(np.random.default_rng(0).choice(A0.shape[1], N_NODES, replace=False)).to(DEV)
        vals = []; ref = None
        for j in range(N_FLOOR + 1):
            C = corr(hidden(m, X[j * N_IMG:(j + 1) * N_IMG])[:, nodes_s]); lcc, e = percolate(C); s = summary(lcc, N_NODES)
            if j == 0:
                ref = e
            vals.append(dict(k_c=s["k_c"], jump=s["max_jump"], giant8=s["lcc_grid"][-1], overlap=None if j == 0 else jaccard(e, ref)))
        rows = (PB[run][arm]["pos"] if arm.startswith("blur_") else PE[run][arm]["pos"]); E = rows[-1]["epoch"]
        P_ = [x for x in rows if x["epoch"] >= E // 4]

        def sd(xs):
            xs = np.array([np.nan if x is None else x for x in xs], dtype=float); return float(np.nanstd(xs)) if np.isfinite(xs).any() else float("nan")
        f = dict(k_c=sd([v["k_c"] for v in vals]), jump=sd([v["jump"] for v in vals]), giant8=sd([v["giant8"] for v in vals]),
                 overlap=float(np.mean([v["overlap"] for v in vals[1:]])))
        pl = dict(k_c=sd([x["k_c"] for x in P_]), jump=sd([x["max_jump"] for x in P_]), giant8=sd([x["lcc_grid"][-1] for x in P_]),
                  overlap=float(np.mean([x["overlap_prev"] for x in P_])))
        RES["floor"][f"{run}/{arm}"] = dict(floor=f, plateau=pl)
        print(f"  {run:14s} {arm:8s} s.d. floor/plateau: k_c {f['k_c']:.2f}/{pl['k_c']:.2f}  jump {f['jump']:.3f}/{pl['jump']:.3f}  "
              f"giant@8 {f['giant8']:.3f}/{pl['giant8']:.3f} | overlap images {f['overlap']:.2f} vs epochs {pl['overlap']:.2f}", flush=True)

json.dump(RES, open(f"{ROOT}/diagnostics/percolation_checks.json", "w"), indent=1)
print("WROTE diagnostics/percolation_checks.json")
