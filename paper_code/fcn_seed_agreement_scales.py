"""Seed-to-seed agreement of FCN hidden maps: what is there before training, what training adds,
and at which spatial scales. Inference only (MNIST, seeds 42-46, with and without LI).

Networks: trained = paper checkpoints; untrained = the same seed's initial weights, rebuilt the way
scripts/run.py does (set_seed(seed), then build_model; checked against the checkpoints' init below
only in so far as the configured init is reproduced).
Each hidden layer's activation (784 units = 28x28 map) is band-passed into radial frequency bands of
the 2-D FFT (cycles per 28 px; edges 0,2,4,6,9,13,20 -> wavelength >14, 7-14, 4.7-7, 3.1-4.7,
2.2-3.1, 1.4-2.2 px) plus the unfiltered map. Per layer and band, over 1000 test images:
  TT   unit-aligned correlation between two TRAINED seeds (10 pairs)
  UU   the same between two UNTRAINED seeds: agreement given by the wiring alone
  TU   trained network vs its OWN initial weights (same seed): how far training moves the maps
  TX   trained map vs the input image (band-passed the same way): how image-like the map is
  UX   untrained map vs the input image
  P    fraction of the map's power in the band (trained), to weigh the bands
Outputs diagnostics/fcn_seed_agreement_scales.{png,json}.
Run: PYTHONPATH=. python paper_code/fcn_seed_agreement_scales.py [r1|r2] [coarse|fine]
  r1 = paper FCN (3x3 Moore neighbourhood), seeds 42-46, with and without LI
  r0 / r2 = FCN with a pointwise gate (no neighbourhood) / a 5x5 neighbourhood (with LI only), seeds 42-44;
  r1 restricted to the same seeds for comparison
  coarse = the 6 bands above; fine = 14 bands (edges 0,2,3,...,12,14,16,20)
"""
from conwaynet import REPO_ROOT
import sys
import itertools
import json
import random

import numpy as np
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from conwaynet.config import load_config
from conwaynet.data import load_dataset
from conwaynet.models import build_model

ROOT = REPO_ROOT; DEV = "cuda"; N = 1000; HW = 28
TAG = sys.argv[1] if len(sys.argv) > 1 else "r1"; RES = sys.argv[2] if len(sys.argv) > 2 else "coarse"
SUFFIX = "" if (TAG, RES) == ("r1", "coarse") else f"_{TAG}_{RES}"
if TAG == "r1":
    SEEDS = {42: "fcn_mnist_tanh_ns_ki_d10", 43: "fcn_mnist_tanh_ns_ki_d10_s43", 44: "fcn_mnist_tanh_ns_ki_d10_s44",
             45: "fcn_mnist_tanh_ns_ki_d10_s45", 46: "fcn_mnist_tanh_ns_ki_d10_s46"}
    MECHS = ("ca", "none")
    if RES == "fine":
        SEEDS = {k: v for k, v in SEEDS.items() if k <= 44}
else:                                       # r0 = pointwise gate (no neighbourhood), r2 = 5x5
    SEEDS = {k: f"fcn_mnist_tanh_ns_ki_d10_{TAG}_s{k}" for k in (42, 43, 44)}
    MECHS = ("ca",)
MAP = json.load(open(f"{ROOT}/paper_runs/_paper_net_configs.json"))
EDGES = [0, 2, 4, 6, 9, 13, 20] if RES == "coarse" else [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 20]
BANDS = [f"{HW / EDGES[i + 1]:.1f}-{(HW / EDGES[i] if EDGES[i] else float('inf')):.1f}px" for i in range(len(EDGES) - 1)]
BANDS[0] = f">{HW / EDGES[1]:.0f}px"


@torch.no_grad()
def hidden(m, X):
    cur = X.view(X.size(0), -1); outs = []
    for i, lin in enumerate(m.linears):
        t = lin(cur); t = m.norms[i](t) if m.norms is not None else t; z = torch.tanh(t)
        if m.gate is not None:
            z = m.gate(cur, z)
        cur = z; outs.append(cur.view(-1, HW, HW))
    return outs


yy, xx = np.meshgrid(np.arange(HW), np.arange(HW), indexing="ij")
R = np.sqrt((yy - HW // 2) ** 2 + (xx - HW // 2) ** 2)
MASKS = [torch.tensor(((R >= EDGES[i]) & (R < EDGES[i + 1])).astype(np.float32), device=DEV) for i in range(len(EDGES) - 1)]


def bands(M):
    """(n,28,28) -> list of band-passed maps (n,28,28), one per band, then the unfiltered map."""
    F = torch.fft.fftshift(torch.fft.fft2(M), dim=(-2, -1))
    return [torch.fft.ifft2(torch.fft.ifftshift(F * k, dim=(-2, -1))).real.cpu() for k in MASKS] + [M.cpu()]   # CPU: the GPU is shared with training


def power(M):
    F = torch.fft.fftshift(torch.fft.fft2(M - M.mean((-2, -1), keepdim=True)), dim=(-2, -1)).abs() ** 2
    tot = F.sum((-2, -1))
    return [float(((F * k).sum((-2, -1)) / tot.clamp_min(1e-12)).mean()) for k in MASKS]


def corr(A, B):
    """Mean over images of the correlation between two (n,28,28) stacks."""
    a = A.flatten(1); b = B.flatten(1); a = a - a.mean(1, keepdim=True); b = b - b.mean(1, keepdim=True)
    return float(((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-8)).mean())


def init_model(cfg, meta, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    return build_model(cfg, meta)


cfg0 = load_config(MAP[SEEDS[42]][0])
_, _, te, meta = load_dataset(cfg0["data_path"], 500, cfg0["data_name"], val_frac=cfg0["val_frac"], split_seed=cfg0["split_seed"],
                              data_norms=cfg0["data_norms"], output_path=None, num_workers=2)
X = []
for xb, _ in te:
    X.append(xb)
    if sum(len(x) for x in X) >= N: break
X = torch.cat(X)[:N].to(DEV)
XB = bands(X[:, 0])

H = {}
for mech in MECHS:
    for sd, run in SEEDS.items():
        cfg = load_config(MAP[run][0 if mech == "ca" else 1])
        m = init_model(cfg, meta, cfg["seed"]).to(DEV).eval()
        H[(mech, "U", sd)] = [bands(h) for h in hidden(m, X)]
        m.load_state_dict(torch.load(f"{ROOT}/paper_runs/{run}/model_{mech}.pth", map_location="cpu")); m.to(DEV).eval()
        tr = hidden(m, X)
        H[(mech, "T", sd)] = [bands(h) for h in tr]
        H[(mech, "P", sd)] = [power(h) for h in tr]
L = len(H[("ca", "T", 42)]); NB = len(BANDS) + 1
pairs = list(itertools.combinations(SEEDS, 2))
res = {}
for mech in MECHS:
    out = {k: np.zeros((L, NB, 2)) for k in ("TT", "UU", "TU", "TX", "UX")}
    for l in range(L):
        for b in range(NB):
            v = {"TT": [corr(H[(mech, "T", a)][l][b], H[(mech, "T", c)][l][b]) for a, c in pairs],
                 "UU": [corr(H[(mech, "U", a)][l][b], H[(mech, "U", c)][l][b]) for a, c in pairs],
                 "TU": [corr(H[(mech, "T", s)][l][b], H[(mech, "U", s)][l][b]) for s in SEEDS],
                 "TX": [corr(H[(mech, "T", s)][l][b], XB[b]) for s in SEEDS],
                 "UX": [corr(H[(mech, "U", s)][l][b], XB[b]) for s in SEEDS]}
            for k in out:
                out[k][l, b] = (np.mean(v[k]), np.std(v[k]))
    P = np.mean([H[(mech, "P", s)] for s in SEEDS], 0)
    res[mech] = {k: out[k].tolist() for k in out} | {"P": P.tolist()}
    print(f"\n{mech}  (rows: layers 1..{L}; cols: bands {BANDS} + unfiltered)")
    for k in ("TT", "UU", "TU", "TX", "UX"):
        print(f" {k}")
        for l in (0, 2, 4, 6, 9):
            print(f"   L{l + 1:2d} " + " ".join(f"{out[k][l, b, 0]:6.3f}" for b in range(NB)))
    print(" P (trained power fraction per band)")
    for l in (0, 2, 4, 6, 9):
        print(f"   L{l + 1:2d} " + " ".join(f"{v:6.3f}" for v in P[l]))
json.dump(dict(bands=BANDS + ["unfiltered"], edges=EDGES, seeds=list(SEEDS), **res),
          open(f"{ROOT}/diagnostics/fcn_seed_agreement_scales{SUFFIX}.json", "w"), indent=2)
if SUFFIX:
    print("WROTE json; heatmap figure only for the default r1/coarse run"); sys.exit(0)

fig, ax = plt.subplots(2, 4, figsize=(18, 7.6), sharex=True, sharey=True)
titles = {"TT": "trained vs trained seed", "UU": "untrained vs untrained seed", "TU": "trained vs its own init", "TX": "trained map vs input image"}
for r, mech in enumerate(("ca", "none")):
    for c, k in enumerate(("TT", "UU", "TU", "TX")):
        A = np.array(res[mech][k])[:, :len(BANDS), 0].T
        im = ax[r][c].imshow(A, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto", origin="lower")
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                ax[r][c].text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color="k" if abs(A[i, j]) < 0.6 else "w")
        ax[r][c].set_title(f"{'with' if mech == 'ca' else 'without'} LI: {titles[k]}", fontsize=10)
        ax[r][c].set_yticks(range(len(BANDS))); ax[r][c].set_yticklabels(BANDS, fontsize=8)
        ax[r][c].set_xticks(range(L)); ax[r][c].set_xticklabels(range(1, L + 1), fontsize=8)
    ax[r][0].set_ylabel("spatial scale (wavelength)")
for c in range(4): ax[1][c].set_xlabel("hidden layer")
fig.colorbar(im, ax=ax, shrink=0.6, label="correlation")
fig.suptitle("FCN on MNIST: agreement of hidden maps by spatial scale (mean over seeds / seed pairs, 1000 test images)", fontsize=12)
fig.savefig(f"{ROOT}/diagnostics/fcn_seed_agreement_scales.png", dpi=130, bbox_inches="tight")
print("WROTE diagnostics/fcn_seed_agreement_scales.png")
