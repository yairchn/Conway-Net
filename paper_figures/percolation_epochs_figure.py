"""Percolation of the co-activation graph over training, one network pair: three panels stacked vertically.
(A) without and (B) with lateral inhibition: giant-component fraction (colour) against the mean degree <k>
(edges added strongest first) and the training epoch, with the transition k_c (giant component = half the
nodes) traced in cyan and the Erdos-Renyi threshold <k> = 1 dashed; (C) the percolation curves at selected
epochs for both networks, with the Erdos-Renyi curve.
Reads diagnostics/percolation_epochs.json (paper_code/percolation_epochs.py).
Run: PYTHONPATH=. python paper_figures/percolation_epochs_figure.py <run> <abs|pos>
"""
from conwaynet import REPO_ROOT
import json
import sys

import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = REPO_ROOT
RUN, KIND = sys.argv[1], sys.argv[2]
d = json.load(open(f"{ROOT}/diagnostics/percolation_epochs.json"))[RUN]
ARMS = [("none", "without lateral inhibition", "#30638e"), ("ca", "with lateral inhibition", "#d1495b")]
EDGE = {"abs": "|correlation|", "pos": "positive correlation"}[KIND]
NET = {"fcn": "FCN", "cnn": "CNN"}[RUN.split("_")[0]]

k = np.array(d["ca"]["k_grid"]); kmax = k[-1]
S = np.ones_like(k)                                      # Erdos-Renyi giant component: S = 1 - exp(-k S)
for _ in range(500):
    S = 1 - np.exp(-k * S)
S = np.where(k > 1, S, 0.0)

fig, axes = plt.subplots(3, 1, figsize=(7.2, 16), gridspec_kw=dict(height_ratios=[1, 1, 1.7]))
for ax, (mech, lab, col) in zip(axes[:2], ARMS):
    r = d[mech]; ep = np.array(r["epochs"]); L = np.array([row["lcc_grid"] for row in r[KIND]])
    im = ax.imshow(L, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1,
                   extent=[k[0], kmax, ep[0] - 0.5, ep[-1] + 0.5], interpolation="nearest")
    kc = np.array([np.nan if row["k_c"] is None else row["k_c"] for row in r[KIND]])
    ax.plot(kc, ep, color="#4fd1ff", lw=1.6, label=r"transition $\langle k\rangle_c$ (giant = half the nodes)")
    ax.axvline(1, color="white", ls="--", lw=1, label=r"Erdős–Rényi $\langle k\rangle_c=1$")
    ax.set_xlim(0, kmax); ax.set_ylabel("training epoch"); ax.set_xlabel(r"mean degree $\langle k\rangle$ (edges added strongest first)")
    ax.set_title(f"{'AB'[ARMS.index((mech, lab, col))]}  {NET} {lab}", loc="left", fontsize=12, weight="bold")
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.85)
    fig.colorbar(im, ax=ax, label="giant-component fraction", pad=0.02)

ax = axes[2]; last = d["ca"]["epochs"][-1]
SEL = [(0, ":"), (3, "--"), (20, "-."), (last, "-")]
for mech, lab, col in ARMS:
    r = d[mech]
    for e, ls in SEL:
        ax.plot(k, r[KIND][r["epochs"].index(e)]["lcc_grid"], ls=ls, color=col, lw=1.8 if e == last else 1.3)
ax.plot(k, S, color="#555", lw=1.4)
h_arm = [Line2D([], [], color=col, lw=2, label=lab) for _, lab, col in ARMS] + [Line2D([], [], color="#555", lw=1.4, label="Erdős–Rényi (theory)")]
h_ep = [Line2D([], [], color="#444", ls=ls, lw=1.4, label=f"epoch {e}") for e, ls in SEL]
ax.axvline(1, color="#999", ls=":", lw=0.8); ax.axhline(0.5, color="#999", ls=":", lw=0.8)
ax.set_xlim(0, kmax); ax.set_ylim(0, 1.02); ax.grid(alpha=0.25)
ax.set_xlabel(r"mean degree $\langle k\rangle$"); ax.set_ylabel("giant-component fraction")
ax.set_title("C  percolation curves at selected epochs", loc="left", fontsize=12, weight="bold")
ax.add_artist(ax.legend(handles=h_arm, fontsize=8.5, loc="upper left", bbox_to_anchor=(0.0, -0.09), frameon=False, handlelength=4))
ax.legend(handles=h_ep, fontsize=8.5, loc="upper right", bbox_to_anchor=(1.0, -0.09), ncol=2, frameon=False, handlelength=5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.suptitle(f"Co-activation percolation over training: {NET}, MNIST\n(edges: {EDGE}, strongest first)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = f"{ROOT}/diagnostics/percolation_epochs_{RUN}_{KIND}.png"; fig.savefig(out, dpi=150)
print("WROTE", out)
