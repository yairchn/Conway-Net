"""Per-image distributions of the signed saliency localisation, log10(on / off), with and without lateral inhibition
(reviewer P2). Reads diagnostics/saliency_signed_perimage.npz (paper_code/saliency_signed_metrics.py); loads no network.
One row per trained network (all seeds pooled, all 1000 images per seed, images excluded by the metric's rules
dropped); > 0 = more saliency per pixel on the object than off it. Text: share of images favouring the object.
Writes paper_figures/out/saliency_signed_distributions.{png,pdf}.
Run: python paper_figures/saliency_signed_distributions.py
"""
from conwaynet import REPO_ROOT
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = REPO_ROOT; CA, NO = "#d1495b", "#30638e"; ZERO_GRAD = 1e-8
D = np.load(f"{ROOT}/diagnostics/saliency_signed_perimage.npz")
ROWS = [("FCN", d) for d in ("mnist", "fashionmnist", "kmnist", "emnist")] + \
       [("CNN", d) for d in ("mnist", "fashionmnist", "kmnist", "emnist")] + \
       [("CNN", d) for d in ("mnist_color", "fashionmnist_color", "kmnist_color", "emnist_color")] + [("WRN", "emnist")]
NAME = {"mnist": "MNIST", "fashionmnist": "Fashion", "kmnist": "KMNIST", "emnist": "EMNIST"}


def values(arch, ds, mech):
    lr, fav = [], []
    for key in D.files:
        a, d, m, v, run, field = key.split("|")
        if (a, d, m, v, field) != (arch, ds, mech, "trained", "on"):
            continue
        p = {f: D[f"{a}|{d}|{m}|{v}|{run}|{f}"] for f in ("on", "off", "gmax", "f")}
        keep = (p["f"] >= 0.03) & (p["f"] <= 0.97) & (p["gmax"] >= ZERO_GRAD)
        ok = keep & (p["on"] > 0) & (p["off"] > 0)
        lr.append(np.log10(p["on"][ok] / p["off"][ok])); fav.append(p["on"][keep] > p["off"][keep])
    return np.concatenate(lr), float(np.concatenate(fav).mean())


fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(ROWS) + 1.0))
labels = []
for r, (arch, ds) in enumerate(ROWS):
    y = len(ROWS) - 1 - r
    for dy, mech, col in ((0.17, "ca", CA), (-0.17, "none", NO)):
        v, fav = values(arch, ds, mech)
        vp = ax.violinplot(v, positions=[y + dy], vert=False, widths=0.32, showextrema=False)
        for b in vp["bodies"]:
            b.set_facecolor(col); b.set_edgecolor("none"); b.set_alpha(0.75)
        q = np.percentile(v, [25, 50, 75])
        ax.plot(q[[0, 2]], [y + dy] * 2, color="#333333", lw=1.2); ax.plot(q[1], y + dy, "o", ms=3, color="white", mec="#333333", mew=0.8)
        ax.text(2.02, y + dy, f"{100 * fav:.0f}%", va="center", fontsize=7, color="#333333")
    colour = ds.endswith("_color")
    labels.append((y, f"{arch.replace('FCN', 'MLP')} {NAME[ds.replace('_color', '')]}" + (" (colour)" if colour else "")))
ax.axvline(0, color="#555555", lw=0.8, ls="--")
ax.set_yticks([y for y, _ in labels]); ax.set_yticklabels([t for _, t in labels], fontsize=8)
ax.set_xlim(-1.0, 2.0); ax.set_xlabel(r"$\log_{10}(\overline{|s|}_\mathrm{on}/\overline{|s|}_\mathrm{off})$ per image  (> 0: object)", fontsize=8.5)
ax.text(2.02, len(ROWS) - 0.35, "favours\nobject", fontsize=7, color="#333333", va="bottom")
ax.tick_params(axis="x", labelsize=8); ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x", alpha=0.25)
ax.plot([], [], color=CA, lw=6, alpha=0.75, label="with lateral inhibition"); ax.plot([], [], color=NO, lw=6, alpha=0.75, label="plain")
ax.legend(fontsize=8, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{ROOT}/paper_figures/out/saliency_signed_distributions.{ext}", dpi=200)
print("WROTE paper_figures/out/saliency_signed_distributions.{png,pdf}")
