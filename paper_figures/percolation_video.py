"""Percolation video matching the paper figure (fig:percolation): for one network pair at one epoch, the
co-activation graph emerges as positive-correlation edges are added strongest first; nodes light up (coloured
by layer) when they join the giant component, and below each graph the giant-component fraction against <k>
builds up in sync, with the Erdos-Renyi curve. Columns: without | with lateral inhibition. Same nodes (random
subsample, seed 0, restricted to live units: s.d. >= SD_MIN), images (first N_IMG test images) and snapshots as paper_code/percolation_epochs.py.
Run: PYTHONPATH=. python paper_figures/percolation_video.py <run> <epoch>
  -> paper_figures/out/percolation_video_<run>_ep<epoch>.gif
"""
from conwaynet import REPO_ROOT
import sys

import numpy as np
import torch
import networkx as nx
import imageio.v2 as imageio
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

sys.path.insert(0, f"{REPO_ROOT}/paper_code")
from percolation_epochs import hidden, corr, N_NODES, N_IMG, K_MAX, SD_MIN, ROOT, DEV  # noqa: E402
from conwaynet.config import load_config  # noqa: E402
from conwaynet.data import load_dataset  # noqa: E402
from conwaynet.models import build_model  # noqa: E402

RUN, EPOCH = sys.argv[1], int(sys.argv[2])
NF = 100; DISP_EDGES = 1500
ARMS = [("none", "without lateral inhibition", "#30638e"), ("ca", "with lateral inhibition", "#d1495b")]


def panel(mech):
    rd = f"{ROOT}/runs/percol_epochs/{RUN}"; cfg = load_config(f"{ROOT}/configs/percol_epochs/{RUN}_{mech}.json")
    _, _, te, meta = load_dataset(cfg["data_path"], 1000, cfg["data_name"], val_frac=cfg["val_frac"], split_seed=cfg["split_seed"],
                                  data_norms=cfg["data_norms"], output_path=rd, num_workers=2)
    X = torch.cat([x for x, _ in te])[:N_IMG].to(DEV)
    m = build_model(cfg, meta).to(DEV).eval()
    m.load_state_dict(torch.load(f"{rd}/snapshots/model_{mech}_ep{EPOCH}.pth", map_location=DEV))
    A = hidden(m, X); nodes = np.random.default_rng(0).choice(A.shape[1], N_NODES, replace=False)
    # layer of each node: FCN layers are equal blocks of the flattened width; CNN stage 1 then stage 2
    if hasattr(m, "linears"):
        layer = nodes // m.linears[0].out_features; n_layers = len(m.linears)
    else:
        with torch.no_grad():
            n1 = m.conv1(X[:1]).numel()                  # stage-1 units come first in hidden()
        layer = (nodes >= n1).astype(int); n_layers = 2
    An = A[:, torch.from_numpy(nodes).to(DEV)]
    live = np.nonzero((An.std(0) >= SD_MIN).cpu().numpy())[0]   # live units only, as in percolation_epochs.graph
    nodes, layer = nodes[live], layer[live]
    C = corr(An[:, torch.from_numpy(live).to(DEV)])
    n = len(live); iu = np.triu_indices(n, 1); w = C[iu]
    order = np.argsort(-w)[: int(K_MAX * n / 2)]; order = order[w[order] > 0]
    E = np.stack([iu[0][order], iu[1][order]], 1)
    G = nx.Graph(); G.add_nodes_from(range(n)); G.add_edges_from(E.tolist())
    pos = nx.spring_layout(G, k=1.3 / np.sqrt(n), iterations=80, seed=0); P = np.array([pos[i] for i in range(n)])
    counts = np.unique(np.linspace(1, len(E), NF).astype(int))
    parent = np.arange(n); size = np.ones(n, np.int64); members, ks, gf = [], [], []

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    done = 0
    for c in counts:
        for a, b in E[done:c]:
            ra, rb = find(a), find(b)
            if ra != rb:
                if size[ra] < size[rb]:
                    ra, rb = rb, ra
                parent[rb] = ra; size[ra] += size[rb]
        done = c
        roots = np.array([find(i) for i in range(n)]); g = np.bincount(roots).argmax()
        members.append(roots == g); ks.append(2 * c / n); gf.append((roots == g).mean())
    return dict(P=P, segs=P[E], counts=counts, ig=np.array(members), ks=np.array(ks), gf=np.array(gf),
                layer=layer, n_layers=n_layers, n=n)


print(f"building {RUN} epoch {EPOCH} ...", flush=True)
PA = {mech: panel(mech) for mech, _, _ in ARMS}
nf = min(len(PA[m]["counts"]) for m in PA)
kgrid = np.linspace(0, K_MAX, 400); S = np.ones_like(kgrid)
for _ in range(500):
    S = 1 - np.exp(-kgrid * S)
S = np.where(kgrid > 1, S, 0.0)

fig = plt.figure(figsize=(11, 8.4)); gs = fig.add_gridspec(2, 2, height_ratios=[2.0, 1.0], hspace=0.12, wspace=0.12)
art = {}
for j, (mech, lab, col) in enumerate(ARMS):
    p = PA[mech]; ag = fig.add_subplot(gs[0, j]); ac = fig.add_subplot(gs[1, j])
    ag.set_xticks([]); ag.set_yticks([]); [s.set_visible(False) for s in ag.spines.values()]
    ag.set_xlim(p["P"][:, 0].min() - .05, p["P"][:, 0].max() + .05); ag.set_ylim(p["P"][:, 1].min() - .05, p["P"][:, 1].max() + .05)
    lc = LineCollection([], colors="#bbbbbb", linewidths=0.3, alpha=0.5); ag.add_collection(lc)
    sc = ag.scatter(p["P"][:, 0], p["P"][:, 1], c="#dddddd", s=9, edgecolors="none", zorder=3)
    ac.plot(kgrid, S, color="#555", lw=1.3, label="Erdős–Rényi")
    ac.plot(p["ks"], p["gf"], color=col, lw=1.0, alpha=0.2)
    grow, = ac.plot([], [], color=col, lw=2.4, label=lab); mark, = ac.plot([], [], "o", color=col, ms=7)
    ac.axvline(1, color="#999", ls=":", lw=0.8); ac.axhline(0.5, color="#999", ls=":", lw=0.8)
    ac.set_xlim(0, K_MAX); ac.set_ylim(0, 1.02); ac.grid(alpha=0.22)
    ac.set_xlabel(r"mean degree $\langle k\rangle$"); ac.set_ylabel("giant-component fraction")
    ac.legend(frameon=False, fontsize=9, loc="upper left")
    for s in ("top", "right"):
        ac.spines[s].set_visible(False)
    art[mech] = dict(lc=lc, sc=sc, grow=grow, mark=mark, ttl=ag.set_title("", fontsize=11), lab=lab)
net = {"fcn": "FCN", "cnn": "CNN"}[RUN.split("_")[0]]
fig.suptitle(f"{net}, MNIST, epoch {EPOCH}: co-activation graph as positive-correlation edges are added strongest first\n"
             "live units only (of 1000 sampled); nodes join the giant component (coloured by layer); a step = a burst of nodes joining", fontsize=11.5)

frames = []
for f in range(nf):
    for mech, _, _ in ARMS:
        p = PA[mech]; a = art[mech]; ig = p["ig"][f]; prev = p["ig"][f - 1] if f else np.zeros_like(ig)
        face = plt.cm.viridis(p["layer"] / max(p["n_layers"] - 1, 1)); face[~ig] = (0.87, 0.87, 0.87, 1.0)
        a["sc"].set_facecolors(face); new = ig & ~prev
        a["sc"].set_sizes(np.where(new, 40, np.where(ig, 13, 8))); a["sc"].set_edgecolors(np.where(new, "#111", "none"))
        a["lc"].set_segments(p["segs"][: min(p["counts"][f], DISP_EDGES)])
        a["grow"].set_data(p["ks"][: f + 1], p["gf"][: f + 1]); a["mark"].set_data([p["ks"][f]], [p["gf"][f]])
        a["ttl"].set_text(f"{a['lab']} ({p['n']} live units)\n⟨k⟩ = {p['ks'][f]:.2f}   giant = {p['gf'][f]:.2f}")
    fig.canvas.draw(); frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
frames += [frames[-1]] * 20                              # hold the last frame
out = f"{ROOT}/paper_figures/out/percolation_video_{RUN}_ep{EPOCH}.gif"
imageio.mimsave(out, frames, duration=0.1, loop=0); plt.close(fig)
print("WROTE", out, flush=True)
