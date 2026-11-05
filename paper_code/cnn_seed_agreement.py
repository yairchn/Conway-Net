"""Does the agreement between independently trained networks (paper Section 4.2, MLP: unit-for-unit correlation of
the hidden maps across seeds) also hold in the CNN? Paper CNN on MNIST, seeds 0-4, with and without LI; inference only.
Per stage (y1: 32 x 26 x 26, y2: 64 x 12 x 12) and seed pair (10 pairs), averaged over 1000 test images:
  same unit      per image, correlation of the two seeds' activations unit for unit (channel c, position p) --
                 the MLP metric; channel order is free in a CNN, so this can be low for equivalent networks
  matched ch.    the same after pairing the channels one-to-one (Hungarian on channel-map correlation over images
                 and positions) -- spatial agreement once the channel order is factored out
  channel mean   per image, correlation of the channel-mean maps (order-free; what the stage-2 gate reads)
  CKA            linear CKA of the (images x units) matrices
Writes diagnostics/cnn_seed_agreement.json.
Run: PYTHONPATH=. python paper_code/cnn_seed_agreement.py
"""
from conwaynet import REPO_ROOT
import itertools
import json

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from conwaynet.config import load_config
from conwaynet.data import load_dataset
from conwaynet.models import build_model

ROOT = REPO_ROOT; DEV = "cuda"; N = 1000
RUNS = ["cnn_mnist_perch"] + [f"cnn_mnist_perch_s{k}" for k in (1, 2, 3, 4)]
MAP = json.load(open(f"{ROOT}/paper_runs/_paper_net_configs.json"))


@torch.no_grad()
def stages(m, x):
    z1 = m.act(m.norm1(m.conv1(x)))
    if m.li1 is not None:
        z1 = m.li1(x, m.occ_in(z1))
    z2 = F.max_pool2d(m.act(m.norm2(m.conv2(z1))), 2)
    if m.li2 is not None:
        z2 = m.li2(m.occ_in(z1), m.occ_in(z2))
    return [z1.cpu().numpy(), z2.cpu().numpy()]


def rowcorr(a, b):
    a = a - a.mean(-1, keepdims=True); b = b - b.mean(-1, keepdims=True)
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-8)


def cka(A, B):
    A = A - A.mean(0); B = B - B.mean(0)
    return float(np.linalg.norm(A.T @ B) ** 2 / (np.linalg.norm(A.T @ A) * np.linalg.norm(B.T @ B)))


def channel_match(A, B):
    """A, B: (N, C, H, W). Permutation of B's channels that best matches A's (correlation over images x positions)."""
    a = A.transpose(1, 0, 2, 3).reshape(A.shape[1], -1); b = B.transpose(1, 0, 2, 3).reshape(B.shape[1], -1)
    a = (a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-8); b = (b - b.mean(1, keepdims=True)) / (b.std(1, keepdims=True) + 1e-8)
    C = np.nan_to_num(a @ b.T / a.shape[1]); r, c = linear_sum_assignment(-C)
    return c


cfg0 = load_config(MAP[RUNS[0]][0])
_, _, te, meta = load_dataset(cfg0["data_path"], 500, cfg0["data_name"], val_frac=cfg0["val_frac"], split_seed=cfg0["split_seed"],
                              data_norms=cfg0["data_norms"], output_path=None, num_workers=2)
X = []
for xb, _ in te:
    X.append(xb)
    if sum(len(x) for x in X) >= N: break
X = torch.cat(X)[:N].to(DEV)

res = {}
for mech in ("ca", "none"):
    acts = []
    for run in RUNS:
        m = build_model(load_config(MAP[run][0 if mech == "ca" else 1]), meta)
        m.load_state_dict(torch.load(f"{ROOT}/paper_runs/{run}/model_{mech}.pth", map_location="cpu")); m.to(DEV).eval()
        acts.append(stages(m, X))
    out = {}
    for s in range(2):
        rows = {k: [] for k in ("same_unit", "matched_channels", "channel_mean", "cka")}
        for i, j in itertools.combinations(range(len(RUNS)), 2):
            A, B = acts[i][s], acts[j][s]
            rows["same_unit"].append(float(rowcorr(A.reshape(N, -1), B.reshape(N, -1)).mean()))
            Bp = B[:, channel_match(A, B)]
            rows["matched_channels"].append(float(rowcorr(A.reshape(N, -1), Bp.reshape(N, -1)).mean()))
            rows["channel_mean"].append(float(rowcorr(A.mean(1).reshape(N, -1), B.mean(1).reshape(N, -1)).mean()))
            rows["cka"].append(cka(A.reshape(N, -1), B.reshape(N, -1)))
        out[f"stage{s + 1}"] = {k: (float(np.mean(v)), float(np.std(v))) for k, v in rows.items()}
    res[mech] = out
    for st, d in out.items():
        print(f"{mech:4s} {st}: " + "  ".join(f"{k} {v[0]:.3f}±{v[1]:.3f}" for k, v in d.items()), flush=True)
json.dump(res, open(f"{ROOT}/diagnostics/cnn_seed_agreement.json", "w"), indent=1)
