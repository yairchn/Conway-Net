"""Geometry of the FCN co-activation graph: does the graph of live units lie in the image plane, and how does the
plain network's graph compare with a random graph of the same size?
Same nodes (random subsample, seed 0), images (first N_IMG test images), live filter and positive edges as
paper_code/percolation_epochs.py, at mean degree K_FIX. Every FCN hidden unit sits at a pixel (unit index mod 784).
Per arm and epoch:
  - giant-component fraction at K_FIX, component sizes, diameter and mean shortest path of the giant
  - the same for a random (Erdos-Renyi) graph with the same number of nodes and edges (N_ER draws)
  - clustering coefficient against the random graph's, effective rank of the correlation matrix
  - spring layout (as in the videos): circular correlation between each giant unit's layout angle and its pixel
    angle about the image centre (max over reflection; chance ~ 1/sqrt(n)), mean pixel distance from the centre
    of giant and non-giant units, and the ink s.d. at those pixels
  - layer counts of giant and non-giant units
Writes diagnostics/percolation_geometry.json and paper_figures/out/percolation_geometry.png (seed 42, last epoch
and epoch 150: layout coloured by pixel angle, and the fraction of units in the giant per pixel).
Run: PYTHONPATH=. python paper_code/percolation_geometry.py
"""
from conwaynet import REPO_ROOT
import json
import sys

import numpy as np
import torch
import networkx as nx
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

sys.path.insert(0, f"{REPO_ROOT}/paper_code")
from percolation_epochs import hidden, corr, N_NODES, N_IMG, SD_MIN, ROOT, DEV  # noqa: E402
from percolation_blur_snapshots import build_blur                               # noqa: E402
from conwaynet.config import load_config                                        # noqa: E402
from conwaynet.data import load_dataset                                         # noqa: E402
from conwaynet.models import build_model                                        # noqa: E402

K_FIX = 4.0; N_ER = 20; EPOCHS = [150, 300]
ARMS = [("fcn_mnist_s42", "ca"), ("fcn_mnist_s43", "ca"), ("fcn_mnist_s44", "ca"),
        ("fcn_mnist_s42", "none"), ("fcn_mnist_s43", "none"), ("fcn_mnist_s44", "none"),
        ("fcn_mnist_s42", "blur_0.9"), ("fcn_mnist_s42", "blur_0.5")]
FIG = [("fcn_mnist_s42", "none"), ("fcn_mnist_s42", "ca"), ("fcn_mnist_s42", "blur_0.9")]
LABEL = {"none": "plain", "ca": "with lateral inhibition", "blur_0.9": "input blur, λ = 0.9", "blur_0.5": "input blur, λ = 0.5"}


def edges_at_k(C, k):
    n = C.shape[0]; iu = np.triu_indices(n, 1); w = C[iu]
    o = np.argsort(-w)[: int(k * n / 2)]; o = o[w[o] > 0]
    return np.stack([iu[0][o], iu[1][o]], 1)


def giant_stats(G):
    comps = sorted(nx.connected_components(G), key=len, reverse=True); Gg = G.subgraph(comps[0])
    return comps, nx.diameter(Gg), nx.average_shortest_path_length(Gg)


def circ(a, b):
    return float(max(abs(np.mean(np.exp(1j * (a - b)))), abs(np.mean(np.exp(1j * (a + b))))))


def model_for(run, arm):
    if arm.startswith("blur_"):
        cfg = load_config(f"{ROOT}/configs/percol_epochs/{run}_ca.json")
        return cfg, lambda c, mt: build_blur(c, mt, float(arm.split("_")[1]))
    return load_config(f"{ROOT}/configs/percol_epochs/{run}_{arm}.json"), build_model


def analyse(run, arm, ep, X, ink_sd, rng):
    cfg, build = model_for(run, arm); rd = f"{ROOT}/runs/percol_epochs/{run}"
    _, _, _, meta = load_dataset(cfg["data_path"], 1000, cfg["data_name"], val_frac=cfg["val_frac"], split_seed=cfg["split_seed"],
                                 data_norms=cfg["data_norms"], output_path=rd, num_workers=2)
    m = build(cfg, meta).to(DEV).eval()
    m.load_state_dict(torch.load(f"{rd}/snapshots/model_{arm}_ep{ep}.pth", map_location=DEV))
    A = hidden(m, X); nodes = np.random.default_rng(0).choice(A.shape[1], N_NODES, replace=False)
    An = A[:, torch.from_numpy(nodes).to(DEV)]; live = np.nonzero((An.std(0) >= SD_MIN).cpu().numpy())[0]; n = len(live)
    C = corr(An[:, torch.from_numpy(live).to(DEV)]); ev = np.linalg.eigvalsh(C + np.eye(n))
    E = edges_at_k(C, K_FIX); G = nx.Graph(); G.add_nodes_from(range(n)); G.add_edges_from(E.tolist())
    comps, diam, aspl = giant_stats(G); ig = np.isin(np.arange(n), list(comps[0]))
    er = [giant_stats(nx.gnm_random_graph(n, len(E), seed=int(rng.integers(1 << 30)))) for _ in range(N_ER)]
    pix = nodes[live] % 784; r, c = pix // 28, pix % 28; lyr = nodes[live] // 784
    pos = nx.spring_layout(G, k=1.3 / np.sqrt(n), iterations=80, seed=0); P = np.array([pos[i] for i in range(n)])
    Pc = P - P[ig].mean(0); lay_ang = np.arctan2(Pc[:, 1], Pc[:, 0]); pix_ang = np.arctan2(r - 13.5, c - 13.5)
    rad = np.hypot(r - 13.5, c - 13.5)
    res = dict(run=run, arm=arm, epoch=ep, n_live=n, n_edges=len(E), giant=float(ig.mean()),
               components=[len(x) for x in comps[:5]], singletons=sum(len(x) == 1 for x in comps),
               diameter=diam, mean_path=aspl,
               er_giant=float(np.mean([len(x[0][0]) / n for x in er])), er_diameter=float(np.mean([x[1] for x in er])),
               er_mean_path=float(np.mean([x[2] for x in er])),
               clustering=nx.average_clustering(G), er_clustering=float(np.mean([nx.average_clustering(nx.gnm_random_graph(n, len(E), seed=s)) for s in range(5)])),
               eff_rank=float(ev.sum() ** 2 / (ev ** 2).sum()),
               angle_corr=circ(lay_ang[ig], pix_ang[ig]), angle_chance=float(1 / np.sqrt(ig.sum())),
               radius_giant=float(rad[ig].mean()), radius_rest=float(rad[~ig].mean()) if (~ig).any() else None,
               ink_sd_giant=float(ink_sd[pix[ig]].mean()), ink_sd_rest=float(ink_sd[pix[~ig]].mean()) if (~ig).any() else None,
               layers_giant=np.bincount(lyr[ig], minlength=10).tolist(), layers_rest=np.bincount(lyr[~ig], minlength=10).tolist())
    print(f"{run} {arm:8s} ep{ep}: live {n}  giant {res['giant']:.2f} (ER {res['er_giant']:.2f})  diameter {diam} (ER {res['er_diameter']:.1f})  "
          f"clustering {res['clustering']:.2f} (ER {res['er_clustering']:.3f})  angle corr {res['angle_corr']:.2f} (chance {res['angle_chance']:.2f})  "
          f"radius giant/rest {res['radius_giant']:.1f}/{res['radius_rest']}", flush=True)
    return res, dict(P=P, ig=ig, pix_ang=pix_ang, r=r, c=c)


def main():
    cfg0 = load_config(f"{ROOT}/configs/percol_epochs/fcn_mnist_s42_ca.json")
    _, _, te, _ = load_dataset(cfg0["data_path"], 1000, cfg0["data_name"], val_frac=cfg0["val_frac"], split_seed=cfg0["split_seed"],
                               data_norms=cfg0["data_norms"], output_path=f"{ROOT}/runs/percol_epochs/fcn_mnist_s42", num_workers=2)
    X = torch.cat([x for x, _ in te])[:N_IMG].to(DEV)
    ink_sd = (X[:, 0].cpu().numpy() * 0.5 + 0.5).reshape(N_IMG, 784).std(0)       # data_norms 0.5/0.5 -> pixel in [0,1]
    rng = np.random.default_rng(1); out = []; keep = {}
    for run, arm in ARMS:
        for ep in EPOCHS:
            res, lay = analyse(run, arm, ep, X, ink_sd, rng); out.append(res); keep[(run, arm, ep)] = lay
    json.dump(out, open(f"{ROOT}/diagnostics/percolation_geometry.json", "w"), indent=1)
    fig, axes = plt.subplots(len(EPOCHS) * 2, len(FIG), figsize=(4.2 * len(FIG), 4.0 * 2 * len(EPOCHS)))
    for j, (run, arm) in enumerate(FIG):
        for i, ep in enumerate(EPOCHS):
            d = keep[(run, arm, ep)]; ig = d["ig"]; ax = axes[2 * i, j]
            ax.scatter(d["P"][~ig, 0], d["P"][~ig, 1], s=5, c="#c8c8c8", edgecolors="none")
            ax.scatter(d["P"][ig, 0], d["P"][ig, 1], s=6, c=d["pix_ang"][ig], cmap="hsv", vmin=-np.pi, vmax=np.pi, edgecolors="none")
            res = next(x for x in out if (x["run"], x["arm"], x["epoch"]) == (run, arm, ep))
            ax.set_title(f"{LABEL[arm]}, epoch {ep}\n{res['n_live']} live units, giant {res['giant']:.2f}, angle corr. {res['angle_corr']:.2f}", fontsize=9.5)
            ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]
            ax = axes[2 * i + 1, j]; g = np.zeros((28, 28)); n = np.zeros((28, 28))
            np.add.at(g, (d["r"], d["c"]), ig); np.add.at(n, (d["r"], d["c"]), 1)
            im = ax.imshow(np.where(n > 0, g / np.maximum(n, 1), np.nan), cmap="viridis", vmin=0, vmax=1)
            ax.set_title("fraction of a pixel's units in the giant", fontsize=9.5); ax.set_xticks([]); ax.set_yticks([])
    fig.subplots_adjust(right=0.9, hspace=0.25); cax = fig.add_axes([0.92, 0.12, 0.012, 0.3])
    fig.colorbar(im, cax=cax, label="fraction in giant (white: no live unit)")
    fig.suptitle(f"FCN seed 42, ⟨k⟩ = {K_FIX:g}: spring layout of the live units, giant component coloured by its pixel's angle about the image centre (grey: not in the giant)", fontsize=10.5)
    fig.savefig(f"{ROOT}/paper_figures/out/percolation_geometry.png", dpi=130, bbox_inches="tight")
    print("WROTE diagnostics/percolation_geometry.json paper_figures/out/percolation_geometry.png")


if __name__ == "__main__":
    main()
