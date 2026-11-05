"""Appendix companion to the noise-resilience figure (paper_noise_resilience.py): the same grid of networks, plotting the
accuracy gap with lateral inhibition minus plain at each severity, so the gap values can be read directly.
Per seed pair (seed k with lateral inhibition against seed k plain) the gap curve is the mean over Gaussian, shot and impulse
noise (the WRN per-type panels: that noise only); line = mean over the five pairs, band and error bars = s.d. over pairs.
Severity 0 is the clean-accuracy gap. Reads paper_runs/_noise_per_type.json (paper_code/noise_per_type.py), the scoring
run every panel of the main figure is taken from.
Run: PYTHONPATH=. python paper_figures/paper_noise_gap.py
"""
from conwaynet import REPO_ROOT
import json

import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = REPO_ROOT; CA, NO = "#d1495b", "#30638e"
PT = json.load(open(f"{ROOT}/paper_runs/_noise_per_type.json"))
NOISE3 = ("gaussian_noise", "shot_noise", "impulse_noise")
GRID = [["FCN2/mnist", "FCN2/fashion", "FCN2/kmnist", "FCN2/emnist"],
        ["CNN/mnist", "CNN/fashion", "CNN/kmnist", "CNN/emnist"],
        ["CNN/mcolor", "CNN/fcolor", "CNN/kcolor", "CNN/ecolor"],
        ["WRN/cifar10", "WRNN/gaussian_noise", "WRNN/shot_noise", "WRNN/impulse_noise"],
        ["GCN/mnist", "GCNN/gaussian_noise", "GCNN/shot_noise", "GCNN/impulse_noise"]]
RUN = {"CNN/mnist": "cnn_mnist_perch", "CNN/fashion": "cnn_fashionmnist_perch", "CNN/kmnist": "cnn_kmnist_perch",
       "CNN/emnist": "cnn_emnist_perch", "CNN/mcolor": "cnn_mnist_color_perch", "CNN/fcolor": "cnn_fashionmnist_color_perch",
       "CNN/kcolor": "cnn_kmnist_color_perch", "CNN/ecolor": "cnn_emnist_color_perch", "WRN/cifar10": "wrn_cifar10"}
SFX = ["", "_s1", "_s2", "_s3", "_s4"]
TITLE = {"mnist": "MNIST", "fashion": "FashionMNIST", "kmnist": "KMNIST", "emnist": "EMNIST", "mcolor": "colour MNIST",
         "fcolor": "colour FashionMNIST", "kcolor": "colour KMNIST", "ecolor": "colour EMNIST", "cifar10": "CIFAR-10"}


def records(key):
    fam, ds = key.split("/")
    if fam == "FCN2":
        return [PT[f"fcn2_matched_{ {'fashion': 'fashionmnist'}.get(ds, ds) }_s{k}|matched"] for k in range(5)], NOISE3
    if fam == "WRNN":
        return [PT[f"wrn_cifar10{s}|equal"] for s in SFX], (ds,)
    if fam in ("GCN", "GCNN"):
        return [PT[f"gcn2_mnist_s{k}|equal"] for k in range(5)], NOISE3 if fam == "GCN" else (ds,)
    return [PT[f"{RUN[key]}{s}|equal"] for s in SFX], NOISE3


fig, axes = plt.subplots(len(GRID), 4, figsize=(13, 13.6), sharey="row")
for r, row in enumerate(GRID):
    for c, key in enumerate(row):
        ax = axes[r][c]; recs, types = records(key)
        G = np.array([np.mean([np.array(rec[f"{t}|ca"]) - np.array(rec[f"{t}|none"]) for t in types], 0) for rec in recs])
        mu, sd = G.mean(0), G.std(0); xs = np.arange(len(mu))
        ax.axvspan(1, 5, color="#e9e4d8", zorder=0, lw=0); ax.axhline(0, color="#555", lw=1)
        ax.plot(xs, mu, "-o", color=CA, lw=2.4, ms=4)
        ax.fill_between(xs, mu - sd, mu + sd, color=CA, alpha=0.3, lw=0)
        ax.errorbar(xs, mu, yerr=sd, fmt="none", ecolor=CA, elinewidth=1.2, capsize=2.5, capthick=1.2)
        fam, ds = key.split("/")
        name = {"FCN2": "MLP-2 matched ", "CNN": "CNN ", "WRN": "WRN ", "GCN": "GCN ", "WRNN": "WRN CIFAR-10 ",
                "GCNN": "GCN MNIST "}[fam] + (ds.split("_")[0] if fam in ("WRNN", "GCNN") else TITLE[ds])
        ax.set_title(name, fontsize=12.5, weight="bold"); ax.set_xlim(0, 9); ax.set_xticks(xs)
        ax.tick_params(labelsize=11); ax.grid(alpha=0.2)
        ax.text(0.97, 0.97, f"$\\Delta_{{1-5}}${mu[1:6].mean():+.1f}\n$\\Delta_{{1-9}}${mu[1:].mean():+.1f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=10.5)
        if c == 0: ax.set_ylabel("gap (points)", fontsize=13)
        if r == len(GRID) - 1: ax.set_xlabel("noise severity", fontsize=13)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); out = f"{ROOT}/paper_figures/out/noise_gap.png"; fig.savefig(out, dpi=135)
print("WROTE", out)
