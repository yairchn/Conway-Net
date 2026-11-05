"""Sensitivity of the coffee-automaton measures (paper_code/fcn_coffee_complexity.py) to their settings: coarse window
(3/5/7), coarse levels (2/3/4), fine levels (4/8/16), compressor (zlib/lzma). MNIST FCN, with lateral inhibition and
plain, 5 seeds, 500 test images; per setting the mean over hidden layers 2-10 (and layer 1) of entropy and apparent
complexity, each normalised by an i.i.d. uniform random map; pure-noise reference = uniform noise on [-1, 1].
Run: PYTHONPATH=. python paper_code/fcn_coffee_sensitivity.py
"""
from conwaynet import REPO_ROOT
import itertools
import json
import lzma
import zlib

import numpy as np
import torch
from scipy import ndimage

from conwaynet.config import load_config
from conwaynet.data import load_dataset
from conwaynet.models import build_model

ROOT = REPO_ROOT; DEV = "cuda"; N = 500; HW = 28
MAP = json.load(open(f"{ROOT}/paper_runs/_paper_net_configs.json"))
RUNS = ["fcn_mnist_tanh_ns_ki_d10"] + [f"fcn_mnist_tanh_ns_ki_d10_s{s}" for s in (43, 44, 45, 46)]
COMP = {"zlib": lambda b: len(zlib.compress(b, 9)), "lzma": lambda b: len(lzma.compress(b, format=lzma.FORMAT_RAW, filters=[{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}]))}
quant = lambda a, L: np.minimum(((np.clip(a, -1, 1) + 1) / 2 * L).astype(np.uint8), L - 1)
rng = np.random.default_rng(0)


@torch.no_grad()
def hidden(m, X):
    cur = X.view(X.size(0), -1); outs = []
    for i, lin in enumerate(m.linears):
        t = lin(cur); t = m.norms[i](t) if m.norms is not None else t; z = torch.tanh(t)
        if m.gate is not None: z = m.gate(cur, z)
        cur = z; outs.append(cur.cpu().numpy().reshape(-1, HW, HW))
    return outs


cfg = load_config(MAP[RUNS[0]][0])
_, _, te, meta = load_dataset(cfg["data_path"], 500, cfg["data_name"], val_frac=cfg["val_frac"], split_seed=cfg["split_seed"],
                              data_norms=cfg["data_norms"], output_path=None, num_workers=2)
X = next(iter(te))[0][:N].to(DEV)
H = {}
for k, mech in ((0, "ca"), (1, "none")):
    H[mech] = []
    for run in RUNS:
        m = build_model(load_config(MAP[run][k]), meta); m.load_state_dict(torch.load(f"{ROOT}/paper_runs/{run}/model_{mech}.pth", map_location="cpu")); m.to(DEV).eval()
        H[mech].append([h[:150] for h in hidden(m, X)])            # 150 images per network keeps the sweep quick
NOISE = rng.uniform(-1, 1, (150, HW, HW))

settings = [("default", 5, 3, 8, "zlib")] + [(f"window {w}", w, 3, 8, "zlib") for w in (3, 7)] + \
           [(f"coarse levels {c}", 5, c, 8, "zlib") for c in (2, 4)] + [(f"fine levels {f}", 5, 3, f, "zlib") for f in (4, 16)] + \
           [("lzma", 5, 3, 8, "lzma")]
out = {}
print(f"{'setting':18s} | entropy L2-10: with / plain / noise | complexity L1: with / plain | complexity L2-10: with / plain / noise")
for name, w, c, f, comp in settings:
    cs = COMP[comp]
    ref_f = np.mean([cs(rng.integers(0, f, (HW, HW), dtype=np.uint8).tobytes()) for _ in range(100)])
    ref_c = np.mean([cs(rng.integers(0, c, (HW, HW), dtype=np.uint8).tobytes()) for _ in range(100)])
    ent = lambda maps: np.mean([cs(quant(a, f).tobytes()) for a in maps]) / ref_f
    cpx = lambda maps: np.mean([cs(quant(ndimage.uniform_filter(a, w, mode="nearest"), c).tobytes()) for a in maps]) / ref_c
    r = {}
    for mech in ("ca", "none"):
        E = np.array([[ent(h) for h in net] for net in H[mech]]); C = np.array([[cpx(h) for h in net] for net in H[mech]])
        r[mech] = dict(ent=float(E[:, 1:].mean()), cpx1=float(C[:, 0].mean()), cpx=float(C[:, 1:].mean()), cpx_sd=float(C[:, 1:].mean(1).std()))
    r["noise"] = dict(ent=float(ent(NOISE)), cpx=float(cpx(NOISE)))
    out[name] = r
    print(f"{name:18s} | {r['ca']['ent']:.2f} / {r['none']['ent']:.2f} / {r['noise']['ent']:.2f}         | {r['ca']['cpx1']:.2f} / {r['none']['cpx1']:.2f}"
          f"              | {r['ca']['cpx']:.2f} / {r['none']['cpx']:.2f} / {r['noise']['cpx']:.2f}", flush=True)
json.dump(out, open(f"{ROOT}/diagnostics/fcn_coffee_sensitivity.json", "w"), indent=1)
