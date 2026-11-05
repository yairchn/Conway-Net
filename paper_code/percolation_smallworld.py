"""Small-world index and degree distribution of the co-activation graphs of paper_code/percolation_epochs.py
(same nodes, images, active-unit filter, positive edges), at the end of training.
Small world: on the graph at mean degree K in KS (edges strongest first), sigma = (C / C_r) / (L / L_r), with C the
average clustering, L the mean shortest path of the giant component, and C_r, L_r those of Erdos-Renyi graphs with
the same nodes and edges (N_ER draws); also L / L_r alone, since a lattice has C / C_r >> 1 but long paths.
Degree distribution: on the graph of correlations above R_DEG, a discrete power-law fit (Clauset et al.,
`powerlaw`) with normalised log-likelihood-ratio tests against lognormal and exponential (R > 0: power law favoured).
Writes diagnostics/percolation_smallworld.json.
Run: PYTHONPATH=. python paper_code/percolation_smallworld.py
"""
from conwaynet import REPO_ROOT
import json
import sys

import networkx as nx
import numpy as np
import powerlaw
import torch

sys.path.insert(0, f"{REPO_ROOT}/paper_code")
from percolation_epochs import corr, SD_MIN, ROOT, DEV            # noqa: E402
from percolation_operating_point import load                      # noqa: E402

KS = [4.0, 8.0]; N_ER = 10; R_DEG = 0.5
ARMS = [(f"fcn_mnist_s{s}", a, 300) for s in (42, 43, 44) for a in ("ca", "none")] + [("fcn_mnist_s42", "blur_0.9", 300)] + \
       [(f"cnn_mnist_s{s}", a, 200) for s in (0, 1, 2) for a in ("ca", "none")]


def cl(G):
    g = G.subgraph(max(nx.connected_components(G), key=len))
    return nx.average_clustering(G), nx.average_shortest_path_length(g)


def main():
    cache = {}; rng = np.random.default_rng(0); out = []
    for run, arm, ep in ARMS:
        An, _ = load(run, arm, ep, cache); live = np.nonzero((An.std(0) >= SD_MIN).cpu().numpy())[0]
        C = corr(An[:, torch.from_numpy(live).to(DEV)]); n = C.shape[0]; iu = np.triu_indices(n, 1); w = C[iu]; o = np.argsort(-w)
        row = dict(run=run, arm=arm, epoch=ep, n_live=n)
        for k in KS:
            sel = o[: int(k * n / 2)]; sel = sel[w[sel] > 0]
            G = nx.Graph(); G.add_nodes_from(range(n)); G.add_edges_from(zip(iu[0][sel].tolist(), iu[1][sel].tolist()))
            c, l = cl(G); rr = [cl(nx.gnm_random_graph(n, G.number_of_edges(), seed=int(rng.integers(1 << 30)))) for _ in range(N_ER)]
            cr, lr = np.mean([x[0] for x in rr]), np.mean([x[1] for x in rr])
            row[f"k{k:g}"] = dict(C=c, C_r=cr, L=l, L_r=lr, C_ratio=c / cr, L_ratio=l / lr, sigma=(c / cr) / (l / lr))
        deg = np.zeros(n, int); s = w > R_DEG; np.add.at(deg, iu[0][s], 1); np.add.at(deg, iu[1][s], 1); deg = deg[deg > 0]
        fit = powerlaw.Fit(deg, discrete=True, verbose=False)
        Rl, pl = fit.distribution_compare("power_law", "lognormal", normalized_ratio=True)
        Re, pe = fit.distribution_compare("power_law", "exponential", normalized_ratio=True)
        row["degree"] = dict(n=int(len(deg)), mean=float(deg.mean()), max=int(deg.max()), alpha=float(fit.power_law.alpha),
                             xmin=int(fit.power_law.xmin), n_tail=int((deg >= fit.power_law.xmin).sum()),
                             R_vs_lognormal=float(Rl), p_vs_lognormal=float(pl), R_vs_exponential=float(Re), p_vs_exponential=float(pe))
        out.append(row); d = row["degree"]
        print(f"{run} {arm:8s}: n {n:4d} | " + " | ".join(
              f"k={k:g}: C/Cr {row[f'k{k:g}']['C_ratio']:.1f} L/Lr {row[f'k{k:g}']['L_ratio']:.2f} sigma {row[f'k{k:g}']['sigma']:.1f}" for k in KS) +
              f" | deg(r>{R_DEG}) mean {d['mean']:.1f} max {d['max']} alpha {d['alpha']:.2f} xmin {d['xmin']} tail {d['n_tail']}"
              f" R_ln {d['R_vs_lognormal']:+.2f} (p {d['p_vs_lognormal']:.2f}) R_exp {d['R_vs_exponential']:+.2f} (p {d['p_vs_exponential']:.2f})", flush=True)
    json.dump(out, open(f"{ROOT}/diagnostics/percolation_smallworld.json", "w"), indent=1)


if __name__ == "__main__":
    main()
