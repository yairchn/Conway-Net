"""How similar are the hidden activations of two independently trained FCNs (different seeds)?

Paper FCN on MNIST (10 hidden layers of 784 units = a 28x28 map), seeds 42-46, with and without LI.
Inference only. Per hidden layer and per seed pair (10 pairs):
  unit-aligned r   Pearson correlation of the two seeds' activations, unit i vs unit i, for one digit
                   (and its mean over 1000 test images). Without LI the units of a layer are
                   interchangeable, so this can be low even for equivalent representations.
  CKA              linear CKA of the (1000 images x 784 units) activation matrices: similarity of the
                   representations, invariant to unit order and to rotations
  matched r        units paired one-to-one by optimal assignment (Hungarian) on their correlation across
                   the 1000 images, then the mean correlation of matched pairs
Outputs diagnostics/fcn_seed_agreement.png (+ _maps.png) and diagnostics/fcn_seed_agreement.json.
Run: PYTHONPATH=. python paper_code/fcn_seed_agreement.py
"""
from conwaynet import REPO_ROOT
import itertools
import json

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from conwaynet.config import load_config
from conwaynet.data import load_dataset
from conwaynet.models import build_model

ROOT = REPO_ROOT; DEV = "cuda"; N = 1000; HW = 28
SEEDS = {42: "fcn_mnist_tanh_ns_ki_d10", 43: "fcn_mnist_tanh_ns_ki_d10_s43", 44: "fcn_mnist_tanh_ns_ki_d10_s44",
         45: "fcn_mnist_tanh_ns_ki_d10_s45", 46: "fcn_mnist_tanh_ns_ki_d10_s46"}
MAP = json.load(open(f"{ROOT}/paper_runs/_paper_net_configs.json"))


@torch.no_grad()
def hidden(m, X):
    cur = X.view(X.size(0), -1); outs = []
    for i, lin in enumerate(m.linears):
        t = lin(cur); t = m.norms[i](t) if m.norms is not None else t; z = torch.tanh(t)
        if m.gate is not None:
            z = m.gate(cur, z)
        cur = z; outs.append(cur.cpu().numpy())
    return outs


def cka(A, B):
    A = A - A.mean(0); B = B - B.mean(0)
    return float(np.linalg.norm(A.T @ B) ** 2 / (np.linalg.norm(A.T @ A) * np.linalg.norm(B.T @ B)))


def corr_cols(A, B):
    """Correlation matrix between the columns (units) of A and of B over the rows (images)."""
    A = (A - A.mean(0)) / (A.std(0) + 1e-8); B = (B - B.mean(0)) / (B.std(0) + 1e-8)
    return A.T @ B / len(A)


def rowcorr(a, b):
    a = a - a.mean(-1, keepdims=True); b = b - b.mean(-1, keepdims=True)
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-8)


cfg0 = load_config(MAP[SEEDS[42]][0])
_, _, te, meta = load_dataset(cfg0["data_path"], 500, cfg0["data_name"], val_frac=cfg0["val_frac"], split_seed=cfg0["split_seed"],
                              data_norms=cfg0["data_norms"], output_path=None, num_workers=2)
X, Y = [], []
for xb, yb in te:
    X.append(xb); Y.append(yb)
    if sum(len(x) for x in X) >= N: break
X, Y = torch.cat(X)[:N].to(DEV), torch.cat(Y)[:N]
DIGIT = 0                                                     # first test image (a 7)

acts = {}
for mech in ("ca", "none"):
    for sd, run in SEEDS.items():
        m = build_model(load_config(MAP[run][0 if mech == "ca" else 1]), meta)
        m.load_state_dict(torch.load(f"{ROOT}/paper_runs/{run}/model_{mech}.pth", map_location="cpu")); m.to(DEV).eval()
        acts[(mech, sd)] = hidden(m, X)
L = len(acts[("ca", 42)])

res = {}
for mech in ("ca", "none"):
    rows = {k: [] for k in ("digit_r", "unit_r", "cka", "matched_r")}
    for a, b in itertools.combinations(SEEDS, 2):
        per = {k: [] for k in rows}
        for l in range(L):
            A, B = acts[(mech, a)][l], acts[(mech, b)][l]
            per["digit_r"].append(float(rowcorr(A[DIGIT], B[DIGIT])))
            per["unit_r"].append(float(rowcorr(A, B).mean()))
            per["cka"].append(cka(A, B))
            Cm = np.nan_to_num(corr_cols(A, B)); r, c = linear_sum_assignment(-Cm)
            per["matched_r"].append(float(Cm[r, c].mean()))
        for k in rows: rows[k].append(per[k])
    res[mech] = {k: (np.mean(v, 0).tolist(), np.std(v, 0).tolist()) for k, v in rows.items()}
    print(mech)
    for k in rows:
        print(f"  {k:10s} " + " ".join(f"{v:6.3f}" for v in res[mech][k][0]))
json.dump(res, open(f"{ROOT}/diagnostics/fcn_seed_agreement.json", "w"), indent=2)

# figure 1: metrics vs layer
names = {"digit_r": "same unit, one digit", "unit_r": "same unit, 1000 images", "matched_r": "best-matched units", "cka": "CKA (order-free)"}
fig, ax = plt.subplots(1, 4, figsize=(17, 3.8), sharey=True)
for a_, k in zip(ax, ("digit_r", "unit_r", "matched_r", "cka")):
    for mech, col, lab in (("ca", "#d1495b", "with LI"), ("none", "#30638e", "without LI")):
        mu, sd = np.array(res[mech][k][0]), np.array(res[mech][k][1]); x = np.arange(1, L + 1)
        a_.plot(x, mu, "-o", color=col, lw=2, ms=4, label=lab); a_.fill_between(x, mu - sd, mu + sd, color=col, alpha=0.15, lw=0)
    a_.set_title(names[k]); a_.set_xlabel("hidden layer"); a_.grid(alpha=0.2); a_.axhline(0, color="#999", lw=0.8)
    for s_ in ("top", "right"): a_.spines[s_].set_visible(False)
ax[0].set_ylabel("agreement between two seeds"); ax[0].legend(frameon=False)
fig.suptitle("FCN on MNIST: how similar are the hidden activations of two independently trained networks? (mean ± sd over 10 seed pairs)", fontsize=11)
fig.tight_layout(); fig.savefig(f"{ROOT}/diagnostics/fcn_seed_agreement.png", dpi=130)

# figure 2: the digit's maps, two seeds, both networks
show = [0, 2, 4, 6, 9]
fig, ax = plt.subplots(4, len(show) + 1, figsize=(2.0 * (len(show) + 1), 8.4))
img = X[DIGIT, 0].cpu().numpy()
for r, (mech, sd) in enumerate((("ca", 42), ("ca", 43), ("none", 42), ("none", 43))):
    ax[r][0].imshow(img, cmap="gray"); ax[r][0].set_ylabel(f"{'with' if mech == 'ca' else 'without'} LI\nseed {sd}", fontsize=10)
    for c, l in enumerate(show, start=1):
        v = acts[(mech, sd)][l][DIGIT].reshape(HW, HW); lim = np.abs(v).max()
        ax[r][c].imshow(v, cmap="RdBu_r", vmin=-lim, vmax=lim)
        if r == 0: ax[r][c].set_title(f"layer {l + 1}")
    for a_ in ax[r]: a_.set_xticks([]); a_.set_yticks([])
for c, l in enumerate(show, start=1):
    for mech, rr in (("ca", 1), ("none", 3)):
        ax[rr][c].set_xlabel(f"r = {rowcorr(acts[(mech, 42)][l][DIGIT], acts[(mech, 43)][l][DIGIT]):.2f}", fontsize=10)
fig.suptitle("The same digit through two seeds (signed activations; r = unit-aligned correlation between seeds)", fontsize=11)
fig.tight_layout(); fig.savefig(f"{ROOT}/diagnostics/fcn_seed_agreement_maps.png", dpi=130)
print("WROTE diagnostics/fcn_seed_agreement{,_maps}.png")
