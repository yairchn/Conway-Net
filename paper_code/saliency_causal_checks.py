"""Reviewer M4: is the object-localised saliency of the LI networks learned reliance, or an artefact of the gate's
own backward route? MNIST, CNN and FCN, with and without LI, 5 seeds each; inference only.

(1) causal test (no gradients): for each test image, k = its number of object pixels; replace k random object
    pixels, or k random background pixels, with uniform noise in [0,1]; report the accuracy on each (10 draws).
    Object mask = pixels > 0.1 dilated by one pixel (as for saliency); background = the rest.
(2) untrained baseline: saliency separation of the same seeds' initial (untrained) networks.
(3) gradient split: saliency separation with the full gradient, with the image gate's route detached (the gate reads
    a stop-gradient copy of the image), and with every gate's route detached (trunk route only). Exact
    decomposition of the trained network's gradient; no retraining.
Saliency: |d logit_y / dx|, separation = max(on/off, off/on) of the mean on the dilated object mask vs off it, as in
the paper; mean over all 1000 test images (and over the correctly classified ones).
Run: PYTHONPATH=. python paper_code/saliency_causal_checks.py
The paper reports the symmetric score only as region separability; the signed metrics, and the vanishing FCN gradient
with every gate route detached (its separation here is computed on rounding noise), are paper_code/saliency_signed_metrics.py.
"""
from conwaynet import REPO_ROOT
import json
import random

import numpy as np
import torch
from scipy import ndimage

import importlib
LIM = importlib.import_module("conwaynet.lateral_inhibition")
from conwaynet.config import load_config
from conwaynet.data import load_dataset
from conwaynet.models import build_model

ROOT = REPO_ROOT; DEV = "cuda"; N = 1000; DRAWS = 10
MAP = json.load(open(f"{ROOT}/paper_runs/_paper_net_configs.json"))
NETS = {"CNN": ["cnn_mnist_perch"] + [f"cnn_mnist_perch_s{k}" for k in (1, 2, 3, 4)],
        "FCN": ["fcn_mnist_tanh_ns_ki_d10"] + [f"fcn_mnist_tanh_ns_ki_d10_s{k}" for k in (43, 44, 45, 46)]}

# --- gate-route detachment: patch LateralInhibitionCA.forward to read a stop-gradient copy of its gate input
_orig_forward = LIM.LateralInhibitionCA.forward
DETACH = {"mode": "none"}                      # none | image | all


def _patched(self, x, z):
    first = getattr(self, "_reads_image", False)
    if DETACH["mode"] == "all" or (DETACH["mode"] == "image" and first):
        x = x.detach()
    return _orig_forward(self, x, z)


LIM.LateralInhibitionCA.forward = _patched

cfg0 = load_config(MAP[NETS["CNN"][0]][0])
_, _, te, meta = load_dataset(cfg0["data_path"], 500, cfg0["data_name"], val_frac=cfg0["val_frac"], split_seed=cfg0["split_seed"],
                              data_norms=cfg0["data_norms"], output_path=None, num_workers=2)
X, Y = [], []
for xb, yb in te:
    X.append(xb); Y.append(yb)
    if sum(len(x) for x in X) >= N: break
X, Y = torch.cat(X)[:N], torch.cat(Y)[:N]
img01 = (X[:, 0] * 0.5 + 0.5).numpy()
MASK = np.stack([ndimage.binary_dilation(i > 0.1, iterations=1) for i in img01])        # object (dilated)
OBJ = img01 > 0.1                                                                        # object pixels (undilated)


def mark_image_gate(m, arch):
    """Flag the gate that reads the image: CNN li1 (stage 1); FCN: the gate is shared, so mark the first call."""
    for g in m.modules():
        if isinstance(g, LIM.LateralInhibitionCA): g._reads_image = False
    if arch == "CNN" and m.li1 is not None:
        m.li1._reads_image = True


class FCNFirstGate:
    """The FCN applies one gate module at every layer; detach only its first call (the one reading the image)."""
    def __init__(self, m): self.m = m; self.count = 0


def saliency(m, arch, mode):
    DETACH["mode"] = mode
    out = np.zeros((N, 28, 28), dtype=np.float32)
    for i in range(0, N, 250):
        x = X[i:i + 250].clone().to(DEV).requires_grad_(True)
        if arch == "FCN" and mode == "image":
            out[i:i + 250] = fcn_image_detached_grad(m, x, Y[i:i + 250])
        else:
            m(x).gather(1, Y[i:i + 250].to(DEV)[:, None]).sum().backward()
            out[i:i + 250] = x.grad.abs().sum(1).cpu().numpy()
    DETACH["mode"] = "none"
    return out


def fcn_image_detached_grad(m, x, y):
    """FCN forward with the first gate reading a stop-gradient copy of the image (later gates unchanged)."""
    cur = x.view(x.size(0), -1)
    for i, lin in enumerate(m.linears):
        t = lin(cur); t = m.norms[i](t) if m.norms is not None else t; z = torch.tanh(t)
        if m.gate is not None:
            gx = cur
            z = _orig_forward(m.gate, gx.detach() if i == 0 else gx, z)
        cur = z
    out = m.classifier(cur)
    out.gather(1, y.to(DEV)[:, None]).sum().backward()
    return x.grad.abs().sum(1).cpu().numpy()


def sep(S, idx):
    r = []
    for k in idx:
        g, mk = S[k], MASK[k]
        on, off = max(float(g[mk].mean()), 1e-12), max(float(g[~mk].mean()), 1e-12)
        r.append(max(on / off, off / on))
    return float(np.mean(r))


@torch.no_grad()
def causal(m):
    rng = np.random.default_rng(0); acc = {"clean": 0.0, "object": [], "background": []}
    Xd = X.to(DEV)
    acc["clean"] = float((m(Xd).argmax(1).cpu() == Y).float().mean())
    for _ in range(DRAWS):
        for where in ("object", "background"):
            Xc = img01.copy()
            for k in range(N):
                obj = np.flatnonzero(OBJ[k]); bg = np.flatnonzero(~MASK[k]); n = len(obj)
                pool = obj if where == "object" else bg
                sel = rng.choice(pool, min(n, len(pool)), replace=False)
                Xc[k].flat[sel] = rng.uniform(0, 1, len(sel))
            xc = torch.tensor((Xc - 0.5) / 0.5)[:, None].float().to(DEV)
            acc[where].append(float((m(xc).argmax(1).cpu() == Y).float().mean()))
    return {k: (float(np.mean(v)) if isinstance(v, list) else v) for k, v in acc.items()}


res = {}
for arch, runs in NETS.items():
    for mech in ("ca", "none"):
        rows = []
        for run in runs:
            cfg = load_config(MAP[run][0 if mech == "ca" else 1])
            random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"]); torch.cuda.manual_seed_all(cfg["seed"])
            m = build_model(cfg, meta).to(DEV).eval(); mark_image_gate(m, arch)
            untrained = sep(saliency(m, arch, "none"), range(N))
            m.load_state_dict(torch.load(f"{ROOT}/paper_runs/{run}/model_{mech}.pth", map_location="cpu")); m.to(DEV).eval(); mark_image_gate(m, arch)
            with torch.no_grad():
                correct = np.flatnonzero((m(X.to(DEV)).argmax(1).cpu() == Y).numpy())
            row = dict(untrained=untrained)
            for mode in (("none", "image", "all") if mech == "ca" else ("none",)):
                S = saliency(m, arch, mode)
                row[f"sep_{mode}_all"] = sep(S, range(N)); row[f"sep_{mode}_correct"] = sep(S, correct)
            row.update(causal(m)); rows.append(row)
        key = f"{arch}|{mech}"
        res[key] = {k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows]))) for k in rows[0]}
        r = res[key]
        print(f"\n{arch} {'with' if mech == 'ca' else 'without'} LI (5 seeds, mean ± s.d.)")
        print(f"  causal: clean {r['clean'][0]*100:.1f} | object pixels noised {r['object'][0]*100:.1f}±{r['object'][1]*100:.1f} | "
              f"same number of background pixels noised {r['background'][0]*100:.1f}±{r['background'][1]*100:.1f}")
        print(f"  saliency separation, trained, all images: full gradient {r['sep_none_all'][0]:.2f}±{r['sep_none_all'][1]:.2f}"
              + (f" | image-gate route detached {r['sep_image_all'][0]:.2f}±{r['sep_image_all'][1]:.2f}"
                 f" | all gate routes detached (trunk only) {r['sep_all_all'][0]:.2f}±{r['sep_all_all'][1]:.2f}" if mech == "ca" else ""))
        print(f"  (correctly classified only: full {r['sep_none_correct'][0]:.2f})   untrained network: {r['untrained'][0]:.2f}±{r['untrained'][1]:.2f}", flush=True)
json.dump(res, open(f"{ROOT}/diagnostics/saliency_causal_checks.json", "w"), indent=2)
print("WROTE diagnostics/saliency_causal_checks.json")
