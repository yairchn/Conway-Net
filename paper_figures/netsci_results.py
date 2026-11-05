"""Paper figure (Results, co-activation graph): FCN, MNIST, positive-correlation edges, live units only.
(A) giant-component fraction against mean degree <k> at the end of training (seed 42): plain, with lateral
inhibition, input blur lambda = 0.9, a two-dimensional random geometric graph and an Erdos-Renyi graph of the same size.
(B) active units (of 1000 sampled) over training, three seeds each. (C, D) spring layout of the graph at <k> = 4,
epoch 300, seed 42, with lateral inhibition and plain; units in the giant component coloured by the angle of their
pixel about the image centre, the others grey. No titles; see the caption.
Reads diagnostics/percolation_epochs.json, percolation_blur.json, percolation_checks.json; recomputes the layouts
with paper_code/percolation_geometry.analyse.
Run: PYTHONPATH=. python paper_figures/netsci_results.py
"""
from conwaynet import REPO_ROOT
import json
import sys

import numpy as np
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

sys.path.insert(0, f"{REPO_ROOT}/paper_code")
from percolation_geometry import analyse                     # noqa: E402
from percolation_epochs import N_IMG                         # noqa: E402
from conwaynet.config import load_config                     # noqa: E402
from conwaynet.data import load_dataset                      # noqa: E402

ROOT = REPO_ROOT; RUN = "fcn_mnist_s42"; EP = 300
PLAIN = "#30638e"; LI = "#d1495b"; BLUR = "#e09f3e"; REF = "#555555"
E = json.load(open(f"{ROOT}/diagnostics/percolation_epochs.json"))
B = json.load(open(f"{ROOT}/diagnostics/percolation_blur.json"))
R = json.load(open(f"{ROOT}/diagnostics/percolation_checks.json"))["spatial"]["references"]


def er_curve(k):
    S = np.ones_like(k)
    for _ in range(500):
        S = 1 - np.exp(-k * S)
    return np.where(k > 1, S, 0.0)


def layout(ax, res, lay):
    P, ig = lay["P"], lay["ig"]
    ax.scatter(P[~ig, 0], P[~ig, 1], s=4, c="#c8c8c8", edgecolors="none")
    ax.scatter(P[ig, 0], P[ig, 1], s=5, c=lay["pix_ang"][ig], cmap="hsv", vmin=-np.pi, vmax=np.pi, edgecolors="none")
    ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]
    ax.text(0.02, 0.02, f"{res['n_live']} active units, angle corr. {res['angle_corr']:.2f}", transform=ax.transAxes, fontsize=9)


def main():
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2), gridspec_kw=dict(width_ratios=[1.15, 1.15, 1, 1]))
    ax = axes[0]; k = np.array(E[RUN]["ca"]["k_grid"])
    for lab, rows, col in (("plain", E[RUN]["none"]["pos"], PLAIN), ("lateral inhibition", E[RUN]["ca"]["pos"], LI),
                           ("blur, λ = 0.9", B[RUN]["blur_0.9"]["pos"], BLUR)):
        last = rows[-1]; ax.plot(k, last["lcc_grid"], color=col, lw=2, label=f"{lab}, $\\langle k\\rangle_c$ = {last['k_c']:.1f}")
    ax.plot(k, R["rgg_2d"]["lcc_grid"], color=REF, ls="--", lw=1.4, label="2-D geometric graph")
    ax.plot(k, er_curve(k), color=REF, ls=":", lw=1.4, label="Erdős–Rényi")
    ax.axhline(0.5, color="#bbbbbb", lw=0.8, zorder=0)
    ax.set_xlabel(r"mean degree $\langle k\rangle$"); ax.set_ylabel("fraction of active units in the giant component")
    ax.set_xlim(0, 8); ax.set_ylim(0, 1.02); ax.legend(fontsize=7.5, frameon=True, framealpha=0.95, edgecolor="none", loc="lower right", handlelength=1.6)
    ax.text(-0.14, 1.02, "A", transform=ax.transAxes, fontsize=13, fontweight="bold")
    ax = axes[1]
    for s in ("42", "43", "44"):
        for mech, col in (("none", PLAIN), ("ca", LI)):
            d = E[f"fcn_mnist_s{s}"][mech]
            ax.plot(d["epochs"], [r["n_live"] for r in d["pos"]], color=col, lw=1.4, alpha=0.9,
                    label=("plain" if mech == "none" else "with lateral inhibition") if s == "42" else None)
    ax.set_yscale("log"); ax.set_xlabel("training epoch"); ax.set_ylabel("active units (of 1000 sampled)")
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    ax.text(-0.14, 1.02, "B", transform=ax.transAxes, fontsize=13, fontweight="bold")
    cfg0 = load_config(f"{ROOT}/configs/percol_epochs/{RUN}_ca.json")
    _, _, te, _ = load_dataset(cfg0["data_path"], 1000, cfg0["data_name"], val_frac=cfg0["val_frac"], split_seed=cfg0["split_seed"],
                               data_norms=cfg0["data_norms"], output_path=f"{ROOT}/runs/percol_epochs/{RUN}", num_workers=2)
    X = torch.cat([x for x, _ in te])[:N_IMG].to("cuda")
    ink_sd = (X[:, 0].cpu().numpy() * 0.5 + 0.5).reshape(N_IMG, 784).std(0)
    for ax, arm, letter in ((axes[2], "ca", "C"), (axes[3], "none", "D")):
        res, lay = analyse(RUN, arm, EP, X, ink_sd, np.random.default_rng(1)); layout(ax, res, lay)
        ax.text(-0.02, 1.02, letter, transform=ax.transAxes, fontsize=13, fontweight="bold")
    fig.subplots_adjust(wspace=0.28)
    fig.savefig(f"{ROOT}/paper_figures/out/netsci_results.png", dpi=170, bbox_inches="tight")
    print("WROTE paper_figures/out/netsci_results.png")


if __name__ == "__main__":
    main()
