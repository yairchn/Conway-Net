"""Which input region matters: the digit, its edge ring, or the far background? MNIST, FCN and CNN, with and without LI,
5 seeds each (paper networks), 1000 test images.

Regions per image: digit = pixels > 0.1; ring = the digit's Moore neighbourhood minus the digit (one-pixel band around
the strokes); far = pixels at least 3 pixels from the digit.
(1) noise by region: in each region, k pixels replaced by uniform noise in [0,1], with k = the size of the smallest of
    the three regions (per image), so the three conditions corrupt the same number of pixels. Reported: accuracy, and
    for the FCN the relative change of every hidden layer ||noisy - clean|| / ||clean||.
(2) saliency by region: mean |d logit_y / dx| per pixel in each region (full gradient).
Run: PYTHONPATH=. python paper_code/noise_by_region.py
"""
from conwaynet import REPO_ROOT
import json

import numpy as np
import torch
from scipy import ndimage

from conwaynet.config import load_config
from conwaynet.data import load_dataset
from conwaynet.models import build_model

ROOT = REPO_ROOT; DEV = "cuda"; N = 1000; DRAWS = 3
MAP = json.load(open(f"{ROOT}/paper_runs/_paper_net_configs.json"))
NETS = {"FCN": ["fcn_mnist_tanh_ns_ki_d10"] + [f"fcn_mnist_tanh_ns_ki_d10_s{k}" for k in (43, 44, 45, 46)],
        "CNN": ["cnn_mnist_perch"] + [f"cnn_mnist_perch_s{k}" for k in (1, 2, 3, 4)]}
REG = ("digit", "ring", "far")

cfg0 = load_config(MAP[NETS["FCN"][0]][0])
_, _, te, meta = load_dataset(cfg0["data_path"], 500, cfg0["data_name"], val_frac=cfg0["val_frac"], split_seed=cfg0["split_seed"],
                              data_norms=cfg0["data_norms"], output_path=None, num_workers=2)
X, Y = [], []
for xb, yb in te:
    X.append(xb); Y.append(yb)
    if sum(len(x) for x in X) >= N: break
X, Y = torch.cat(X)[:N], torch.cat(Y)[:N]
img01 = (X[:, 0] * 0.5 + 0.5).numpy()
DIG = img01 > 0.1
D1 = np.stack([ndimage.binary_dilation(d, iterations=1) for d in DIG])
D3 = np.stack([ndimage.binary_dilation(d, iterations=2) for d in DIG])          # within 2 px; far = outside
MASKS = {"digit": DIG, "ring": D1 & ~DIG, "far": ~D3}
rng = np.random.default_rng(0)
NOISY = {}
for reg in REG:
    sets = []
    for _ in range(DRAWS):
        Xc = img01.copy()
        for k in range(N):
            kk = min(MASKS[r][k].sum() for r in REG); pool = np.flatnonzero(MASKS[reg][k])
            sel = rng.choice(pool, kk, replace=False); Xc[k].flat[sel] = rng.uniform(0, 1, kk)
        sets.append(torch.tensor((Xc - 0.5) / 0.5)[:, None].float())
    NOISY[reg] = sets


@torch.no_grad()
def fcn_hidden(m, Xs):
    cur = Xs.view(Xs.size(0), -1); outs = []
    for i, lin in enumerate(m.linears):
        t = lin(cur); t = m.norms[i](t) if m.norms is not None else t; z = torch.tanh(t)
        if m.gate is not None: z = m.gate(cur, z)
        cur = z; outs.append(cur.cpu())
    return outs


res = {}
for arch, runs in NETS.items():
    for mech in ("ca", "none"):
        rows = []
        for run in runs:
            m = build_model(load_config(MAP[run][0 if mech == "ca" else 1]), meta)
            m.load_state_dict(torch.load(f"{ROOT}/paper_runs/{run}/model_{mech}.pth", map_location="cpu")); m.to(DEV).eval()
            row = {}
            with torch.no_grad():
                row["acc_clean"] = float((m(X.to(DEV)).argmax(1).cpu() == Y).float().mean())
                for reg in REG:
                    row[f"acc_{reg}"] = float(np.mean([(m(s.to(DEV)).argmax(1).cpu() == Y).float().mean().item() for s in NOISY[reg]]))
            if arch == "FCN":
                Hc = fcn_hidden(m, X.to(DEV))
                for reg in REG:
                    ch = np.mean([[float(((n - c).norm(dim=1) / c.norm(dim=1)).mean()) for c, n in zip(Hc, fcn_hidden(m, s.to(DEV)))]
                                  for s in NOISY[reg]], 0)
                    row[f"change_{reg}"] = ch.tolist()
            G = np.zeros((N, 28, 28), np.float32)
            for i in range(0, N, 250):
                x = X[i:i + 250].clone().to(DEV).requires_grad_(True)
                m(x).gather(1, Y[i:i + 250].to(DEV)[:, None]).sum().backward(); G[i:i + 250] = x.grad.abs().sum(1).cpu().numpy()
            for reg in REG:
                row[f"sal_{reg}"] = float(np.mean([G[k][MASKS[reg][k]].mean() for k in range(N)]))
            tot = sum(row[f"sal_{r}"] for r in REG)
            for reg in REG: row[f"salfrac_{reg}"] = row[f"sal_{reg}"] / tot
            rows.append(row)
        key = f"{arch}|{mech}"; res[key] = {}
        for k in rows[0]:
            v = np.array([r[k] for r in rows]); res[key][k] = (v.mean(0).tolist(), v.std(0).tolist())
        r = res[key]
        print(f"\n{arch} {'with' if mech == 'ca' else 'without'} LI (5 seeds)")
        print(f"  accuracy: clean {r['acc_clean'][0]*100:.1f} | noise on digit {r['acc_digit'][0]*100:.1f}±{r['acc_digit'][1]*100:.1f} | "
              f"edge ring {r['acc_ring'][0]*100:.1f}±{r['acc_ring'][1]*100:.1f} | far background {r['acc_far'][0]*100:.1f}±{r['acc_far'][1]*100:.1f}")
        print("  saliency per pixel (share of the three regions' means): " + " | ".join(f"{reg} {r[f'salfrac_{reg}'][0]*100:.0f}%" for reg in REG))
        if arch == "FCN":
            for reg in REG:
                print(f"  relative change per layer, noise on {reg:5s}: " + " ".join(f"{v:.3f}" for v in r[f"change_{reg}"][0]))
json.dump(res, open(f"{ROOT}/diagnostics/noise_by_region.json", "w"), indent=2)
print("WROTE diagnostics/noise_by_region.json")
