"""Paper figure: percolation of the co-activation graph over training, FCN (left column) and CNN (right column),
one seed each (the paper networks' seeds), positive-correlation edges. Rows: giant-component fraction against
mean degree <k> and epoch without (top) and with (middle) lateral inhibition, k_c traced in cyan and <k> = 1
dashed, the number of live units (of 1000 sampled) at the end of training noted; percolation curves at selected epochs with the Erdos-Renyi curve (bottom). No titles; see the caption.
Reads diagnostics/percolation_epochs.json (paper_code/percolation_epochs.py).
Run: PYTHONPATH=. python paper_figures/percolation_epochs_pair.py
"""
from conwaynet import REPO_ROOT
import json

import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = REPO_ROOT; KIND = "pos"
D = json.load(open(f"{ROOT}/diagnostics/percolation_epochs.json"))
COLS = ["fcn_mnist_s42", "cnn_mnist_s0"]
ARMS = [("none", "without lateral inhibition", "#30638e"), ("ca", "with lateral inhibition", "#d1495b")]
LS = [":", "--", "-.", "-"]

fig, axes = plt.subplots(3, 2, figsize=(11, 13), gridspec_kw=dict(height_ratios=[1, 1, 1.5]))
for c, run in enumerate(COLS):
    d = D[run]; k = np.array(d["ca"]["k_grid"]); kmax = k[-1]
    for r, (mech, _, _) in enumerate(ARMS):
        ax = axes[r, c]; ep = np.array(d[mech]["epochs"]); L = np.array([row["lcc_grid"] for row in d[mech][KIND]])
        im = ax.imshow(L, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1,
                       extent=[k[0], kmax, ep[0] - 0.5, ep[-1] + 0.5], interpolation="nearest")
        kc = np.array([np.nan if row["k_c"] is None else row["k_c"] for row in d[mech][KIND]])
        ax.plot(kc, ep, color="#4fd1ff", lw=1.4); ax.axvline(1, color="white", ls="--", lw=1)
        ax.text(0.03, 0.96, f"{d[mech][KIND][-1]['n_live']} live units at the end", transform=ax.transAxes, ha="left", va="top",
                color="white", fontsize=10)
        ax.set_xlim(0, kmax); ax.set_ylabel("training epoch"); ax.set_xlabel(r"mean degree $\langle k\rangle$")
    S = np.ones_like(k)                                  # Erdos-Renyi giant component: S = 1 - exp(-k S)
    for _ in range(500):
        S = 1 - np.exp(-k * S)
    S = np.where(k > 1, S, 0.0)
    ax = axes[2, c]; last = d["ca"]["epochs"][-1]; sel = [0, 3, 20, last]
    for mech, _, col in ARMS:
        for e, ls in zip(sel, LS):
            ax.plot(k, d[mech][KIND][d[mech]["epochs"].index(e)]["lcc_grid"], ls=ls, color=col, lw=1.8 if e == last else 1.3)
    ax.plot(k, S, color="#555", lw=1.4)
    ax.axvline(1, color="#999", ls=":", lw=0.8); ax.axhline(0.5, color="#999", ls=":", lw=0.8)
    ax.set_xlim(0, kmax); ax.set_ylim(0, 1.02); ax.grid(alpha=0.25)
    ax.set_xlabel(r"mean degree $\langle k\rangle$"); ax.set_ylabel("giant-component fraction")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    h_ep = [Line2D([], [], color="#444", ls=ls, lw=1.4, label=f"epoch {e}") for e, ls in zip(sel, LS)]
    ax.legend(handles=h_ep, fontsize=9, loc="lower right", handlelength=5, frameon=False)
for i, ax in enumerate(axes.T.flatten()):
    ax.text(-0.13, 1.02, "ABCDEF"[i], transform=ax.transAxes, fontsize=14, weight="bold", va="bottom")
h_arm = [Line2D([], [], color=col, lw=2, label=lab) for _, lab, col in ARMS] + [Line2D([], [], color="#555", lw=1.4, label="Erdős–Rényi")]
fig.legend(handles=h_arm, loc="lower center", ncol=3, fontsize=10, frameon=False, handlelength=3)
fig.tight_layout(rect=[0, 0.03, 0.92, 1])
cax = fig.add_axes([0.93, 0.47, 0.015, 0.48]); fig.colorbar(im, cax=cax, label="giant-component fraction")
fig.savefig(f"{ROOT}/paper_figures/out/percolation_epochs.png", dpi=150)
print("WROTE paper_figures/out/percolation_epochs.png")
