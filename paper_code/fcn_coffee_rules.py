"""Coffee-automaton entropy and apparent complexity (Aaronson et al. 2014) of the FCN's hidden maps across the ten
rules of the 300-epoch sweep and the plain network; MNIST, five seeds (42-46) per rule, 500 test images. Measures and settings as
paper_code/fcn_coffee_complexity.py (fixed-range quantisation on [-1, 1]; fine 8 levels; coarse 5x5 mean, 3 levels;
zlib; normalised by an i.i.d. random map). Also the other faithful settings of the sensitivity sweep (window 3 and 7).
Run: PYTHONPATH=. python paper_code/fcn_coffee_rules.py
"""
from conwaynet import REPO_ROOT
import json
import zlib

import numpy as np
import torch
from scipy import ndimage

from conwaynet.config import load_config
from conwaynet.data import load_dataset
from conwaynet.models import build_model

ROOT = REPO_ROOT; DEV = "cuda"; N = 500; HW = 28
MAP = json.load(open(f"{ROOT}/paper_runs/_paper_net_configs.json"))
RULES = ["B5S3", "B5S23", "B3S3", "B3S23", "B34S34", "B36S23", "B3678S34678", "B45678S2345", "B2S", "B1357S1357", "none"]
SEEDS = (42, 43, 44, 45, 46)
run_of = lambda t, k=42: ("fcn_mnist_tanh_ns_ki_d10" + ("" if k == 42 else f"_s{k}")) if t in ("B5S3", "none") else f"fcn_mnist_tanh_ns_ki_d10_{t}_s{k}"
quant = lambda a, L: np.minimum(((np.clip(a, -1, 1) + 1) / 2 * L).astype(np.uint8), L - 1)
cs = lambda q: len(zlib.compress(q.tobytes(), 9))
rng = np.random.default_rng(0)
REF = {L: np.mean([cs(rng.integers(0, L, (HW, HW), dtype=np.uint8)) for _ in range(200)]) for L in (3, 8)}


@torch.no_grad()
def hidden(m, X):
    cur = X.view(X.size(0), -1); outs = []
    for i, lin in enumerate(m.linears):
        t = lin(cur); t = m.norms[i](t) if m.norms is not None else t; z = torch.tanh(t)
        if m.gate is not None: z = m.gate(cur, z)
        cur = z; outs.append(cur.cpu().numpy().reshape(-1, HW, HW))
    return outs


cfg = load_config(MAP[run_of("B5S3")][0])
_, _, te, meta = load_dataset(cfg["data_path"], 500, cfg["data_name"], val_frac=cfg["val_frac"], split_seed=cfg["split_seed"],
                              data_norms=cfg["data_norms"], output_path=None, num_workers=2)
X = next(iter(te))[0][:N].to(DEV)
out = {}
print(f"{'rule':14s} | entropy, layers 2-10 (mean ± s.d. over seeds) | apparent complexity, layers 2-10")
for t in RULES:
    E, C = [], []
    for k in SEEDS:
        run = run_of(t, k); m = build_model(load_config(MAP[run][1 if t == "none" else 0]), meta)
        m.load_state_dict(torch.load(f"{ROOT}/paper_runs/{run}/model_{'none' if t == 'none' else 'ca'}.pth", map_location="cpu")); m.to(DEV).eval()
        H = hidden(m, X)
        E.append(np.mean([np.mean([cs(quant(a, 8)) for a in h]) / REF[8] for h in H[1:]]))
        C.append(np.mean([np.mean([cs(quant(ndimage.uniform_filter(a, 5, mode="nearest"), 3)) for a in h]) / REF[3] for h in H[1:]]))
    out[t] = dict(entropy=(float(np.mean(E)), float(np.std(E))), complexity=(float(np.mean(C)), float(np.std(C))))
    print(f"{t:14s} | {np.mean(E):.3f} ± {np.std(E):.3f} | {np.mean(C):.3f} ± {np.std(C):.3f}", flush=True)
inp = X.cpu().numpy().reshape(-1, HW, HW); out["input"] = dict(entropy=float(np.mean([cs(quant(a, 8)) for a in inp]) / REF[8]))
print("input digit entropy", round(out["input"]["entropy"], 3))
json.dump(out, open(f"{ROOT}/diagnostics/fcn_coffee_rules.json", "w"), indent=1)
