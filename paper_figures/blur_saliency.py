"""Appendix A.6: saliency of the paper CNN (MNIST) with LI vs rule-free controls with the same wiring.

Arms: ca / none = paper checkpoints (paper_runs/cnn_mnist_perch); blur_0.9, blur_0.95, dog_sm =
paper_code/cnn_controls.py (paper_runs/cnn_mnist_perch_controls). Gradient saliency
g = |d logit_y / dx| (channel-summed); on/off-digit separation max(on/off, off/on) with the
Moore-dilated digit mask -- the paper's definition (saliency_onoff_separation.py). Reported as the
mean over 300 correctly classified test images per arm, plus the 10 per-digit samples shown.
Outputs paper_figures/out/blur_saliency_cnn.png + paper_runs/cnn_mnist_perch_controls/_saliency.json.
Run: PYTHONPATH=. python paper_figures/blur_saliency.py
"""
from conwaynet import REPO_ROOT
import importlib.util
import json

import numpy as np
import torch
from scipy import ndimage
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from conwaynet.config import load_config
from conwaynet.data import load_dataset

ROOT = REPO_ROOT; DEV = "cuda"; N = 300
spec = importlib.util.spec_from_file_location("cnn_controls", f"{ROOT}/paper_code/cnn_controls.py")
cc = importlib.util.module_from_spec(spec); spec.loader.exec_module(cc)
ARMS = ["ca", "none", "blur_0.9", "blur_0.95", "dog_sm"]
LABEL = {"ca": "with lateral\ninhibition", "none": "plain", "blur_0.9": "blur λ=0.9", "blur_0.95": "blur λ=0.95", "dog_sm": "DoG-SM"}

cfg = load_config(cc.CFG)
tr, va, te, meta = load_dataset(cfg["data_path"], 256, cfg["data_name"], val_frac=cfg["val_frac"],
                                split_seed=cfg["split_seed"], data_norms=cfg["data_norms"], output_path=None, num_workers=2)
models = {}
for a in ARMS:
    m = cc.build(a, meta, cfg)
    if a not in ("ca", "none"):
        m.load_state_dict(torch.load(f"{cc.OUT}/model_{a}.pth", map_location="cpu"))
    models[a] = m.to(DEV).eval()

X, Y = [], []
for xb, yb in te:
    X.append(xb); Y.append(yb)
    if sum(len(x) for x in X) >= 4000: break
X, Y = torch.cat(X), torch.cat(Y)
img01 = (X * 0.5 + 0.5).clamp(0, 1)[:, 0].numpy()
masks = np.stack([ndimage.binary_dilation(i > 0.1, iterations=1) for i in img01])


def saliency(m, x, y):
    x = x.clone().to(DEV).requires_grad_(True)
    m(x).gather(1, y.to(DEV)[:, None]).sum().backward()
    return x.grad.abs().sum(1).cpu().numpy()


def sep(g, mask):
    on, off = max(float(g[mask].mean()), 1e-9), max(float(g[~mask].mean()), 1e-9)
    return max(on / off, off / on)


def log_ratio(g, mask):
    """Signed: log10(on/off) > 0 favours the digit, < 0 the background (reviewer P2; sep cannot tell them apart)."""
    return float(np.log10(g[mask].mean() / g[~mask].mean()))


digit_idx = [int((Y == d).nonzero()[0]) for d in range(10)]
out = {}
fig, axes = plt.subplots(len(ARMS), 10, figsize=(14, 1.5 * len(ARMS)), squeeze=False)
for r, a in enumerate(ARMS):
    m = models[a]
    with torch.no_grad():
        ok = (m(X.to(DEV)).argmax(1).cpu() == Y).nonzero()[:, 0][:N]
    g = saliency(m, X[ok], Y[ok])
    s300 = [sep(gi, masks[i]) for gi, i in zip(g, ok.tolist())]
    l300 = [log_ratio(gi, masks[i]) for gi, i in zip(g, ok.tolist())]
    gd = saliency(m, X[digit_idx], Y[digit_idx])
    sd = [sep(gi, masks[i]) for gi, i in zip(gd, digit_idx)]
    out[a] = dict(sep_mean300=float(np.mean(s300)), log_ratio_mean300=float(np.mean(l300)), sep_median300=float(np.median(s300)), sep_digits=[float(v) for v in sd])
    res = json.load(open(f"{ROOT}/paper_runs/cnn_mnist_perch/results_{a}.json" if a in ("ca", "none") else f"{cc.OUT}/results_{a}.json"))
    out[a].update(test_clean_acc=res["test_clean_acc"], noise_mCA=res["noise_mCA"])
    print(f"{a:10s} sep mean300 {np.mean(s300):6.2f}  median {np.median(s300):6.2f}  10-digit mean {np.mean(sd):6.2f}  "
          f"clean {res['test_clean_acc']:.4f}  noise {res['noise_mCA']:.4f}", flush=True)
    for d in range(10):
        ax = axes[r][d]; ax.imshow(gd[d], cmap="hot"); ax.set_xticks([]); ax.set_yticks([])
        if r == 0: ax.set_title(str(d), fontsize=10)
    axes[r][0].set_ylabel(f"{LABEL[a]}\nlog$_{{10}}$ on/off {np.mean(l300):+.2f}", fontsize=9, rotation=90, labelpad=2)
json.dump(out, open(f"{cc.OUT}/_saliency.json", "w"), indent=2)
fig.suptitle("Gradient saliency |d logit/dx| per digit — paper CNN with lateral inhibition vs rule-free gates (MNIST); log$_{10}$ on/off = mean over 300 test images (> 0: digit)", fontsize=10.5)
fig.tight_layout(); fig.savefig(f"{ROOT}/paper_figures/out/blur_saliency_cnn.png", dpi=140)
print("WROTE paper_figures/out/blur_saliency_cnn.png")
