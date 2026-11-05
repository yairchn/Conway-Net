"""Two panels (FCN, CNN), with and without LI together: for an image of '1', p(d|x)/p(1|x) for every digit,
after temperature calibration (each network's logits divided by its temperature T, fitted on its validation split;
Guo et al. 2017), with '1' itself as the reference at 10^0. x = rank (0 = '1', then the most likely alternative, ...);
each point is labelled with its digit, since the two networks order the digits differently. Geometric mean over
all test '1's; band = s.d. over 5 seeds.
Reads diagnostics/one_vs_digit_leaderboard.json (paper_code/one_vs_digit_leaderboard.py).
Run: PYTHONPATH=. python paper_figures/one_vs_digit_panels_absolute.py
"""
from conwaynet import REPO_ROOT
import json

import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = REPO_ROOT
d = json.load(open(f"{ROOT}/diagnostics/one_vs_digit_leaderboard.json")); OTHERS = d["digits"]
ARMS = [("ca", "with lateral inhibition", "#d1495b", 0), ("none", "without lateral inhibition", "#30638e", 1)]
import matplotlib.transforms as mtransforms

fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
for ax, arch in zip(axes, ("FCN", "CNN")):
    ax.plot([0], [1.0], "s", color="#222", ms=9, zorder=5)
    ax.annotate("1", (0, 1.0), textcoords="offset points", xytext=(9, 0), ha="left", va="center", fontsize=12, weight="bold", color="#222")
    curves = {}
    for mech, lab, col, side in ARMS:
        g = np.array(d[f"{arch}|{mech}"]["gap_cal_mean"]); G = np.array(d[f"{arch}|{mech}"]["gap_cal_seeds"]); order = np.argsort(g)
        L = -G[:, order] / np.log(10); mu, sd = L.mean(0), L.std(0); x = np.arange(1, len(order) + 1)
        ax.plot([0, 1], [1.0, 10 ** mu[0]], "--", color=col, lw=1.4)
        ax.plot(x, 10 ** mu, "-o", color=col, lw=2.4, ms=6, label=lab)
        ax.fill_between(x, 10 ** (mu - sd), 10 ** (mu + sd), color=col, alpha=0.18, lw=0)
        curves[mech] = (x, mu, sd, order, col)
    # digit labels beside their OWN curve: at each rank the higher curve is labelled above its point, the lower below
    (xa, ma, _, oa, ca_), (xb, mb, _, ob, cb_) = curves["ca"], curves["none"]
    for k in range(len(xa)):
        hi_is_ca = ma[k] >= mb[k]
        for mu_, order_, col_, above in ((ma, oa, ca_, hi_is_ca), (mb, ob, cb_, not hi_is_ca)):
            ax.annotate(str(OTHERS[order_[k]]), (xa[k], 10 ** mu_[k]), textcoords="offset points", xytext=(0, 7 if above else -7),
                        ha="center", va="bottom" if above else "top", fontsize=12, weight="bold", color=col_)
    ax.set_yscale("log")
    ax.set_xticks(range(0, 10)); ax.set_xticklabels([str(k) for k in range(1, 11)])
    ax.set_xlabel("rank by probability (1 = most likely, the correct answer '1')")
    ax.set_title(arch.replace("FCN", "MLP"), fontsize=13, weight="bold"); ax.grid(axis="y", alpha=0.25)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
lo = min(np.min(-np.array(d[k]["gap_cal_mean"])) for k in d if "|" in k) / np.log(10)    # one range for both panels
for ax in axes: ax.set_ylabel("p(digit) / p(1), calibrated"); ax.set_ylim(10 ** (lo - 0.8), 10 ** 0.6)
axes[0].legend(frameon=False, fontsize=10, loc="lower left")
fig.suptitle("For an image of '1': how likely each digit is relative to the correct answer '1' (square), after temperature calibration; "
             "digits ordered by likelihood for each network\n(geometric mean over all 1135 test '1's, mean ± s.d. over 5 seeds)", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.9]); fig.savefig(f"{ROOT}/paper_figures/out/perdigit_probability.png", dpi=150)
print("WROTE paper_figures/out/perdigit_probability.png")
