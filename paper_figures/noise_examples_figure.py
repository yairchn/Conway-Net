"""Appendix figure: one test image per family under gaussian, shot and impulse noise at severities 0 (clean)
to 9, exactly as the networks see it (conwaynet.corruptions, the paper's 9-level ladders). Grayscale images
are corrupted on their single channel (one channel of the three-channel corruption), as in the evaluation code.
Run: PYTHONPATH=. python paper_figures/noise_examples_figure.py
"""
from conwaynet import REPO_ROOT
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from conwaynet import corruptions as C
from conwaynet.eval import _raw_eval_images

ROOT = REPO_ROOT
NOISE = [("gaussian_noise", "Gaussian"), ("shot_noise", "shot"), ("impulse_noise", "impulse")]
SETS = [("MNIST", "./data", "MNIST", 0), ("MNIST_COLOR", "./data", "MNIST (colour)", 1), ("CIFAR10", "/data", "CIFAR-10", 3)]

fig, axes = plt.subplots(len(SETS) * 3, 10, figsize=(13, 1.42 * len(SETS) * 3))
for b, (name, root, title, idx) in enumerate(SETS):
    imgs, _, ch = _raw_eval_images(root, name, idx + 1)
    a = imgs[idx]
    for r, (nm, lab) in enumerate(NOISE):
        row = b * 3 + r
        for lvl in range(10):
            img = a if lvl == 0 else C.apply(a, nm, lvl - 1, np.random.RandomState(1000 * r + lvl))
            ax = axes[row][lvl]
            if ch == 1:
                ax.imshow(img[..., 0].astype(np.float32) / 255.0, cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title("clean" if lvl == 0 else f"severity {lvl}", fontsize=9)
            if lvl == 0:
                ax.set_ylabel(f"{title}\n{lab}" if r == 1 else lab, fontsize=9)
fig.tight_layout(h_pad=0.25, w_pad=0.2)
fig.savefig(f"{ROOT}/paper_figures/out/noise_examples.png", dpi=150, bbox_inches="tight")
print("WROTE paper_figures/out/noise_examples.png")
