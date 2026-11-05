"""Paper figure: which part of the agreement between independently trained FCNs is inherited from the
architecture and which is learned, by spatial scale and depth, for three gate neighbourhoods.

MNIST FCN with LI (10 hidden layers of 28x28), seeds 42-44 per neighbourhood: pointwise gate (r0, the
gate reads the cell's own occupancy), 3x3 Moore (r1, the paper network), 5x5 (r2). For each hidden
layer the maps of two seeds are band-passed by spatial frequency and correlated unit by unit over
1000 test images (fcn_seed_agreement_scales.py, 'fine' bands). Solid: two trained networks; dashed:
the same two seeds' untrained initial networks. Grey: the network without LI (trained), as a floor.
Left: early layers 1-5; right: deep layers 6-10 (mean over layers).
Run: PYTHONPATH=. python paper_figures/fcn_seed_agreement_paper_figure.py
"""
from conwaynet import REPO_ROOT
import json

import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = REPO_ROOT; HW = 28
RUNS = [("r0", "pointwise gate", "#e09f3e"), ("r1", "3×3 neighbourhood (paper)", "#d1495b"), ("r2", "5×5 neighbourhood", "#30638e")]
GROUPS = [("early layers (1–5)", slice(0, 5)), ("deep layers (6–10)", slice(5, 10))]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.9), sharey=True)
for ax, (gname, sl) in zip(axes, GROUPS):
    for tag, lab, col in RUNS:
        d = json.load(open(f"{ROOT}/diagnostics/fcn_seed_agreement_scales_{tag}_fine.json")); E = d["edges"]
        w = np.array([np.sqrt(HW / E[i + 1] * (HW / E[i] if E[i] else HW)) for i in range(len(E) - 1)])
        for k, ls, extra in (("TT", "-", "trained"), ("UU", "--", "untrained")):
            A = np.array(d["ca"][k])[sl, :len(w), 0]
            ax.plot(w, A.mean(0), ls, color=col, lw=2.2 if k == "TT" else 1.6, marker="o" if k == "TT" else None, ms=4,
                    label=f"{lab}, {extra}")
        if tag == "r1":
            A = np.array(d["none"]["TT"])[sl, :len(w), 0]
            ax.plot(w, A.mean(0), "-", color="#999", lw=1.4, label="plain, trained")
    for x, t in ((1, ""), (3, "3 px"), (5, "5 px")):
        if t:
            ax.axvline(x, color="#222", ls=":", lw=1); ax.text(x * 1.03, 0.97, t, fontsize=9)
    ax.axhline(0, color="#bbb", lw=0.8)
    ax.set_xscale("log"); ax.set_xticks([2, 3, 4, 6, 10, 20]); ax.set_xticklabels(["2", "3", "4", "6", "10", "20"]); ax.minorticks_off()
    ax.set_xlabel("spatial scale: wavelength (px, log)"); ax.set_title(gname, fontsize=12); ax.grid(alpha=0.2)
    ax.set_ylim(-0.15, 1.02)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axes[0].set_ylabel("correlation between two seeds")
h, l = axes[1].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=9, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=[0, 0.13, 1, 1]); fig.savefig(f"{ROOT}/paper_figures/out/fcn_seed_agreement.png", dpi=160)
print("WROTE paper_figures/out/fcn_seed_agreement.png")
