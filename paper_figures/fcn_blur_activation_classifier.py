"""Appendix figure: the counterpart of Figure 5 (paper_code/fcn_depth_activation_classifier.py) for the trained blur
control, lambda = 0.95, seed 0 (paper_runs/fcn_mnist_controls/blur_0.95_s0, built with paper_code/fcn_controls.build).
Row 1: hidden layers 1-10 for the same digit-2 test image as Figure 5 (first digit 2 in test order), shown as the
occupancy (y+1)/2 like the lateral-inhibition row of Figure 5 (the blur's background also sits at y = -1). Row 2: the
classifier weight map of each class. One signed colour scale per row (white = 0). Also prints the LFPR (radius 0.25) of
the class-averaged last-hidden maps and of the classifier weights, over the first 4000 test images, for this network.
Writes paper_figures/out/fcn_blur_activation_classifier.png. Run: PYTHONPATH=. python paper_figures/fcn_blur_activation_classifier.py
"""
from conwaynet import REPO_ROOT
import sys

import numpy as np
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from conwaynet.config import load_config
from conwaynet.data import load_dataset

sys.path.insert(0, f"{REPO_ROOT}/paper_code")
import fcn_controls

ROOT = REPO_ROOT; HW = 28; DIGIT = 2; N = 4000; ARM, SEED = "blur_0.95", 0


def lfpr(maps, rf=0.25):
    yy, xx = np.meshgrid(np.arange(HW) - HW // 2, np.arange(HW) - HW // 2, indexing="ij")
    r = np.sqrt(yy ** 2 + xx ** 2); r = r / r.max(); out = []
    for mp in maps:
        P = np.abs(np.fft.fftshift(np.fft.fft2(mp - mp.mean()))) ** 2
        out.append(P[r <= rf].sum() / max(P.sum(), 1e-12))
    return float(np.mean(out))


c = load_config(f"{ROOT}/configs/paper/tanh/fcn_mnist_tanh_ns_ki_ca.json")
_, _, te, meta = load_dataset(c["data_path"], 500, c["data_name"], val_frac=c["val_frac"], split_seed=c["split_seed"],
                              data_norms=c["data_norms"], output_path=None, num_workers=2)
m, _ = fcn_controls.build(ARM, SEED, meta)
m.load_state_dict(torch.load(f"{fcn_controls.OUT}/{ARM}_s{SEED}/model.pth", map_location="cpu")); m.eval()
X, Y = [], []
for xb, yb in te:
    X.append(xb); Y.append(yb)
    if sum(len(x) for x in X) >= N: break
X, Y = torch.cat(X)[:N], torch.cat(Y)[:N]


@torch.no_grad()
def hidden(x):
    cur = x.flatten(1); out = []
    for lin, norm in zip(m.linears, m.norms):
        z = torch.tanh(norm(lin(cur))); cur = m.gate(cur, z); out.append(cur)
    return out


i = int((Y == DIGIT).nonzero()[0][0])
layers = np.stack([h[0].reshape(HW, HW).numpy() for h in hidden(X[i:i + 1])])
W = m.classifier.weight.detach().numpy().reshape(-1, HW, HW)
last = hidden(X)[-1].numpy(); yv = Y.numpy()
act_l = lfpr(np.stack([last[yv == k].mean(0).reshape(HW, HW) for k in range(10)])); cls_l = lfpr(W)
with torch.no_grad():
    acc = float((m(X).argmax(1) == Y).float().mean())
print(f"blur lambda 0.95 s0: test acc (first {N}) {acc * 100:.1f}%, activation LFPR {act_l:.3f}, classifier LFPR {cls_l:.3f}, "
      f"prediction on the digit-2 image {int(m(X[i:i + 1]).argmax())}")

rows = [((layers + 1.0) / 2.0, "blur\n$\\lambda=0.95$\n(occupancy)"), (W, "blur\n$\\lambda=0.95$")]
fig = plt.figure(figsize=(14, 3.9))
blocks = [fig.add_gridspec(1, 10, left=0.09, right=0.95, top=0.80, bottom=0.53, wspace=0.06),
          fig.add_gridspec(1, 10, left=0.09, right=0.95, top=0.33, bottom=0.06, wspace=0.06)]
for r, (maps, lab) in enumerate(rows):
    vm = float(np.abs(maps).max()) or 1.0; axs = []
    for k in range(10):
        ax = fig.add_subplot(blocks[r][0, k]); axs.append(ax)
        im = ax.imshow(maps[k], cmap="RdBu_r", vmin=-vm, vmax=vm); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.set_title(f"layer {k + 1}" if r == 0 else f"class {k}", fontsize=10.5)
        if k == 0: ax.text(-0.12, 0.5, lab, transform=ax.transAxes, ha="right", va="center", fontsize=10.5, weight="bold")
    cb = fig.colorbar(im, ax=axs, fraction=0.012, pad=0.006); cb.ax.tick_params(labelsize=8)
fig.text(0.52, 0.94, f"Activation maps of layers 1-10 for one digit-{DIGIT} test image", ha="center", fontsize=12.5, weight="bold")
fig.text(0.52, 0.45, "Classifier weights", ha="center", fontsize=12.5, weight="bold")
fig.savefig(f"{ROOT}/paper_figures/out/fcn_blur_activation_classifier.png", dpi=150)
print("WROTE paper_figures/out/fcn_blur_activation_classifier.png")
