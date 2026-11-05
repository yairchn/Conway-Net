"""Companion to paper_figures/percolation_video.py, along the other axis of the paper figure: the mean degree is
held fixed at K_FIX and the training epoch advances. At every epoch the co-activation graph keeps its strongest
K_FIX*N/2 positive-correlation edges among its live units (s.d. >= SD_MIN); live nodes are drawn (grey), those in the giant component coloured by layer.
The layout of each epoch starts from the previous one, so the structure moves continuously. Below: the
giant-component fraction at K_FIX and the validation accuracy against epoch. Columns: without | with lateral
inhibition. Same nodes, images and snapshots as paper_code/percolation_epochs.py.
Run: PYTHONPATH=. python paper_figures/percolation_epoch_video.py <run> <k_fix>
  -> paper_figures/out/percolation_epoch_video_<run>_k<k_fix>.gif
"""
from conwaynet import REPO_ROOT
import re
import sys

import numpy as np
import torch
import networkx as nx
import imageio.v2 as imageio
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

sys.path.insert(0, f"{REPO_ROOT}/paper_code")
from percolation_epochs import hidden, corr, N_NODES, N_IMG, SD_MIN, ROOT, DEV  # noqa: E402
from conwaynet.config import load_config  # noqa: E402
from conwaynet.data import load_dataset  # noqa: E402
from conwaynet.models import build_model  # noqa: E402

RUN, K_FIX = sys.argv[1], float(sys.argv[2])
ARMS = [("none", "without lateral inhibition", "#30638e"), ("ca", "with lateral inhibition", "#d1495b")]
DISP_EDGES = 1500


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


def arm(mech):
    rd = f"{ROOT}/runs/percol_epochs/{RUN}"; cfg = load_config(f"{ROOT}/configs/percol_epochs/{RUN}_{mech}.json")
    _, _, te, meta = load_dataset(cfg["data_path"], 1000, cfg["data_name"], val_frac=cfg["val_frac"], split_seed=cfg["split_seed"],
                                  data_norms=cfg["data_norms"], output_path=rd, num_workers=2)
    X = torch.cat([x for x, _ in te])[:N_IMG].to(DEV)
    m = build_model(cfg, meta).to(DEV).eval(); n = N_NODES
    val = {int(e): float(v) for e, v in re.findall(r"epoch +(\d+)/\d+ +train_loss=[\d.]+ +val_acc=([\d.]+)", open(f"{rd}/train_{mech}.log").read())}
    nodes = None; pos = None; out = dict(P=[], segs=[], ig=[], live=[], gf=[], nl=[], epochs=[], val=[])
    for ep in range(cfg["num_epochs"] + 1):
        m.load_state_dict(torch.load(f"{rd}/snapshots/model_{mech}_ep{ep}.pth", map_location=DEV))
        A = hidden(m, X)
        if nodes is None:
            nodes = np.random.default_rng(0).choice(A.shape[1], n, replace=False)
            if hasattr(m, "linears"):
                layer, n_layers = nodes // m.linears[0].out_features, len(m.linears)
            else:
                with torch.no_grad():
                    n1 = m.conv1(X[:1]).numel()
                layer, n_layers = (nodes >= n1).astype(int), 2
        An = A[:, torch.from_numpy(nodes).to(DEV)]
        live = np.nonzero((An.std(0) >= SD_MIN).cpu().numpy())[0]   # live units only, as in percolation_epochs.graph
        nl = len(live); C = corr(An[:, torch.from_numpy(live).to(DEV)]); iu = np.triu_indices(nl, 1); w = C[iu]
        o = np.argsort(-w)[:int(K_FIX * nl / 2)]; o = o[w[o] > 0]; El = np.stack([iu[0][o], iu[1][o]], 1); E = live[El]
        G = nx.Graph(); G.add_nodes_from(live.tolist()); G.add_edges_from(E.tolist())
        init = None if pos is None else {i: pos[i] for i in live if i in pos}
        rest = [i for i in live if pos is None or i not in pos]   # units that came alive: start them at random
        if init is not None and rest:
            init.update({i: np.random.default_rng(i).random(2) for i in rest})
        pos = {**(pos or {}), **nx.spring_layout(G, k=1.3 / np.sqrt(nl), pos=init, iterations=80 if init is None else 15, seed=0)}
        P = np.array([pos.get(i, (np.nan, np.nan)) for i in range(n)]); isl = np.zeros(n, bool); isl[live] = True
        ig = np.zeros(n, bool); ig[live] = giant(nl, El)
        out["P"].append(P); out["segs"].append(P[E[:DISP_EDGES]]); out["ig"].append(ig); out["live"].append(isl)
        out["gf"].append(ig.sum() / nl); out["nl"].append(nl)
        out["epochs"].append(ep); out["val"].append(val.get(ep, np.nan))
        if ep % 50 == 0:
            print(f"  {mech} ep{ep}: giant {ig.mean():.2f}", flush=True)
    out.update(layer=layer, n_layers=n_layers)
    return out


print(f"building {RUN}, <k> = {K_FIX} ...", flush=True)
PA = {mech: arm(mech) for mech, _, _ in ARMS}
nf = len(PA["ca"]["epochs"])

fig = plt.figure(figsize=(11, 8.6)); gs = fig.add_gridspec(2, 2, height_ratios=[2.0, 1.0], hspace=0.14, wspace=0.36)
art = {}
for j, (mech, lab, col) in enumerate(ARMS):
    p = PA[mech]; ag = fig.add_subplot(gs[0, j]); ac = fig.add_subplot(gs[1, j])
    ag.set_xticks([]); ag.set_yticks([]); [s.set_visible(False) for s in ag.spines.values()]
    lc = LineCollection([], colors="#bbbbbb", linewidths=0.3, alpha=0.5); ag.add_collection(lc)
    sc = ag.scatter(p["P"][0][:, 0], p["P"][0][:, 1], c="#dddddd", s=9, edgecolors="none", zorder=3)
    ep = np.array(p["epochs"]); ac.plot(ep, p["gf"], color=col, lw=1.0, alpha=0.2)
    grow, = ac.plot([], [], color=col, lw=2.2, label=f"giant component at ⟨k⟩ = {K_FIX:g}"); mark, = ac.plot([], [], "o", color=col, ms=6)
    ac2 = ac.twinx(); ac2.plot(ep, p["val"], color="#888", lw=1.0, ls="--", label="validation accuracy")
    ac2.set_ylim(0.9, 1.0); ac2.set_ylabel("validation accuracy", color="#777")
    ac.axhline(0.5, color="#999", ls=":", lw=0.8); ac.set_xlim(0, ep[-1]); ac.set_ylim(0, 1.02); ac.grid(alpha=0.22)
    ac.set_xlabel("training epoch"); ac.set_ylabel("giant-component fraction")
    h1, l1 = ac.get_legend_handles_labels(); h2, l2 = ac2.get_legend_handles_labels()
    ac.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8.5, loc="lower right")
    for s in ("top",):
        ac.spines[s].set_visible(False)
    art[mech] = dict(ag=ag, lc=lc, sc=sc, grow=grow, mark=mark, ttl=ag.set_title("", fontsize=11), lab=lab)
net = {"fcn": "FCN", "cnn": "CNN"}[RUN.split("_")[0]]
fig.suptitle(f"{net}, MNIST: the co-activation graph at fixed mean degree ⟨k⟩ = {K_FIX:g} over training\n"
             "live units grey (dead units hidden); giant component coloured by layer; giant fraction over live units", fontsize=11.5)

frames = []
for f in range(nf):
    for mech, _, _ in ARMS:
        p = PA[mech]; a = art[mech]; P = p["P"][f]; ig = p["ig"][f]
        a["sc"].set_offsets(np.nan_to_num(P))
        face = plt.cm.viridis(p["layer"] / max(p["n_layers"] - 1, 1)); face[~ig] = (0.82, 0.82, 0.82, 1.0)
        a["sc"].set_facecolors(face); a["sc"].set_sizes(np.where(ig, 12, np.where(p["live"][f], 7, 0)))
        a["lc"].set_segments(p["segs"][f])
        Q = P[p["live"][f]]; a["ag"].set_xlim(Q[:, 0].min() - .05, Q[:, 0].max() + .05); a["ag"].set_ylim(Q[:, 1].min() - .05, Q[:, 1].max() + .05)
        a["grow"].set_data(p["epochs"][: f + 1], p["gf"][: f + 1]); a["mark"].set_data([p["epochs"][f]], [p["gf"][f]])
        a["ttl"].set_text(f"{a['lab']}   epoch {p['epochs'][f]}\n{p['nl'][f]} live units   giant = {p['gf'][f]:.2f}")
    fig.canvas.draw(); frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
frames += [frames[-1]] * 15
out = f"{ROOT}/paper_figures/out/percolation_epoch_video_{RUN}_k{K_FIX:g}.gif"
imageio.mimsave(out, frames, duration=0.08, loop=0); plt.close(fig)
print("WROTE", out, flush=True)
