"""Figure 13 (decoy patch), 5 seeds, from diagnostics/decoy_patch_check.json (paper_code/decoy_patch_check.py).
Patch sides 2-10 px (decoy regime). Left: predictions flipped vs patch side. Right: share of integrated-gradient magnitude on the patch, with the patch-area chance line.
Run: python paper_figures/decoy_patch_figure.py
"""
from conwaynet import REPO_ROOT
import json

import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = REPO_ROOT; CA, NO = "#d1495b", "#30638e"
d = json.load(open(f"{ROOT}/diagnostics/decoy_patch_check.json")); S = sorted(int(s) for s in d["ca"])
g = lambda mech, k: (np.array([d[mech][str(s)][k][0] for s in S]) * 100, np.array([d[mech][str(s)][k][1] for s in S]) * 100)
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
for mech, col, lab in (("ca", CA, "with lateral inhibition"), ("none", NO, "plain")):
    for a, k, ls, l in ((ax[0], "flip", "-o", lab), (ax[1], "mag", "-o", lab)):
        mu, sd = g(mech, k); a.plot(S, mu, ls, color=col, lw=2.2, label=l); a.fill_between(S, mu - sd, mu + sd, color=col, alpha=0.25, lw=0)
ax[1].plot(S, [s * s / 784 * 100 for s in S], ":", color="#888", label="patch area (chance)")
ax[0].set_ylabel("predictions flipped (%)"); ax[1].set_ylabel("IG magnitude on patch (%)")
ax[0].set_title("predictions flipped by the patch"); ax[1].set_title("attribution on the patch")
for a in ax:
    a.set_xlabel("patch side (px)"); a.set_xticks(S); a.grid(alpha=0.2); a.legend(frameon=False, fontsize=8.5)
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(f"{ROOT}/paper_figures/out/decoy_patch_5seed.png", dpi=150); print("WROTE paper_figures/out/decoy_patch_5seed.png")
