"""Appendix noise breakdowns, one 4x4 figure per network family (the WRN's breakdown is in the main noise figure):
  FCN-10 (equal budget), FCN-2 (matched clean accuracy), CNN grayscale, CNN colour.
Columns: the four datasets (MNIST, FashionMNIST, KMNIST, EMNIST; blue/gold colour versions for the colour CNN).
Rows: Gaussian, shot and impulse noise on the 9-severity ladder (the three types the main figure averages), and the
uniform-noise blend x_t = (1-t) x + t U[0,1] over the fraction t in [0, 1]. Mean +- s.d. over five seeds, with and
without lateral inhibition. Reads paper_runs/_noise_per_type.json and paper_runs/_noise_uniform_blend.json
(paper_code/noise_per_type.py). Writes paper_figures/out/noise_breakdown_<family>.png.
Run: python paper_figures/noise_breakdown_figures.py
"""
from conwaynet import REPO_ROOT
import json

import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = REPO_ROOT; CA, NO = "#d1495b", "#30638e"
PT = json.load(open(f"{ROOT}/paper_runs/_noise_per_type.json"))
UB = json.load(open(f"{ROOT}/paper_runs/_noise_uniform_blend.json"))
DS = ("mnist", "fashionmnist", "kmnist", "emnist"); PRETTY = dict(mnist="MNIST", fashionmnist="FashionMNIST", kmnist="KMNIST", emnist="EMNIST")
FAMILIES = {   # family -> (title, dataset -> list of cache keys (one per seed))
    "fcn10": ("MLP-10, equal training budget",
              {d: [f"fcn_{d}_tanh_ns_ki_d10{s}|equal" for s in ("", "_s43", "_s44", "_s45", "_s46")] for d in DS}),
    "fcn2": ("MLP-2, matched clean accuracy",
             {d: [f"fcn2_matched_{d}_s{k}|matched" for k in range(5)] for d in DS}),
    "cnn_gray": ("CNN, grayscale", {d: [f"cnn_{d}_perch{s}|equal" for s in ("", "_s1", "_s2", "_s3", "_s4")] for d in DS}),
    "cnn_colour": ("CNN, colour", {d: [f"cnn_{d}_color_perch{s}|equal" for s in ("", "_s1", "_s2", "_s3", "_s4")] for d in DS}),
}
ROWS = (("gaussian_noise", "Gaussian"), ("shot_noise", "shot"), ("impulse_noise", "impulse"), ("uniform_blend", "uniform blend"))

for fam, (ftitle, keys) in FAMILIES.items():
    fig, axes = plt.subplots(4, 4, figsize=(13, 11.2))
    for c, d in enumerate(DS):
        for r, (nm, rname) in enumerate(ROWS):
            ax = axes[r][c]; src = UB if nm == "uniform_blend" else PT
            uni = nm == "uniform_blend"
            xs = np.linspace(0, 1, 11) if uni else np.arange(10)
            if not uni:
                ax.axvspan(1, 5, color="#e9e4d8", zorder=0, lw=0)          # standard ImageNet-C severities 1-5
            mu = {}
            for mech, col, lab in (("ca", CA, "with lateral inhibition"), ("none", NO, "plain")):
                cur = np.array([src[k][f"{nm}|{mech}"] for k in keys[d]]); mu[mech] = cur.mean(0); sdv = cur.std(0)
                ax.plot(xs, mu[mech], "-o", color=col, lw=2.2, ms=3.5, label=lab)
                ax.fill_between(xs, mu[mech] - sdv, mu[mech] + sdv, color=col, alpha=0.3, lw=0)
            if uni:
                dall = np.mean(mu["ca"][1:]) - np.mean(mu["none"][1:])
                ax.text(0.04, 0.05, f"$\\Delta${dall:+.1f}", transform=ax.transAxes, ha="left", va="bottom", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
                        fontsize=12, weight="bold", color=(CA if dall >= 0 else NO))
                ax.set_xlim(0, 1); ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
            else:
                d5 = np.mean(mu["ca"][1:6]) - np.mean(mu["none"][1:6]); d9 = np.mean(mu["ca"][1:]) - np.mean(mu["none"][1:])
                ax.text(0.04, 0.17, f"$\\Delta_{{1-5}}${d5:+.1f}", transform=ax.transAxes, ha="left", va="bottom", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
                        fontsize=12, weight="bold", color=(CA if d5 >= 0 else NO))
                ax.text(0.04, 0.05, f"$\\Delta_{{1-9}}${d9:+.1f}", transform=ax.transAxes, ha="left", va="bottom", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
                        fontsize=10, color=(CA if d9 >= 0 else NO))
                ax.set_xlim(0, 9); ax.set_xticks(range(10))
            ax.set_ylim(0, 100); ax.tick_params(labelsize=11); ax.grid(alpha=0.2)
            for s in ("top", "right"): ax.spines[s].set_visible(False)
            if r == 0: ax.set_title(PRETTY[d] + (" (colour)" if fam == "cnn_colour" else ""), fontsize=14, weight="bold")
            if c == 0: ax.set_ylabel(f"{rname}\naccuracy (%)", fontsize=13)
            if r == 2: ax.set_xlabel("noise severity", fontsize=12)
            if r == 3: ax.set_xlabel("noise fraction $t$", fontsize=12)
            if (r, c) == (0, 0): ax.legend(fontsize=10.5, frameon=False, loc="center right")
    fig.suptitle(ftitle, fontsize=15, weight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out = f"{ROOT}/paper_figures/out/noise_breakdown_{fam}.png"; fig.savefig(out, dpi=135); plt.close(fig)
    print("WROTE", out)
