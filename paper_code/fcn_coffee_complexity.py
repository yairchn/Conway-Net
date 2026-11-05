"""Coffee-automaton complexity (Aaronson, Carroll & Ouellette 2014) of the FCN's hidden maps across depth.
MNIST FCN with lateral inhibition and plain, 5 seeds (paper networks, seeds 42-46), first 500 test images.
For each map (the input and hidden layers 1-10, each a 28x28 grid):
  entropy     zlib-compressed size of the fine-grained map: the map quantised to 8 equal levels on the fixed range [-1, 1]
  complexity  zlib-compressed size of the coarse-grained map: 5x5 sliding mean (edge-replicated), quantised to 3 equal levels
              on [-1, 1] (the paper's three fixed buckets)
Both are divided by the compressed size of an i.i.d. uniform random map with the same number of levels (1 = as
incompressible as noise). References: the raw digit (input) and pure noise (i.i.d. uniform on [-1, 1]).
Run: PYTHONPATH=. python paper_code/fcn_coffee_complexity.py
"""
from conwaynet import REPO_ROOT
import json
import zlib

import numpy as np
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import ndimage

from conwaynet.config import load_config
from conwaynet.data import load_dataset
from conwaynet.models import build_model

ROOT = REPO_ROOT; DEV = "cuda"; N = 500; HW = 28; FINE, COARSE, WIN = 8, 3, 5
MAP = json.load(open(f"{ROOT}/paper_runs/_paper_net_configs.json"))
RUNS = ["fcn_mnist_tanh_ns_ki_d10"] + [f"fcn_mnist_tanh_ns_ki_d10_s{s}" for s in (43, 44, 45, 46)]
CA, NO = "#d1495b", "#30638e"


def quant(a, levels):
    """Quantise on the fixed activation range [-1, 1] (as the paper's fixed cream-fraction buckets)."""
    return np.minimum(((np.clip(a, -1, 1) + 1) / 2 * levels).astype(np.uint8), levels - 1)


def csize(q):
    return len(zlib.compress(q.tobytes(), 9))


rng = np.random.default_rng(0)
REF = {L: np.mean([csize(rng.integers(0, L, (HW, HW), dtype=np.uint8)) for _ in range(200)]) for L in (FINE, COARSE)}


def measures(maps):
    """maps: (n, 28, 28) -> (entropy, complexity), each normalised and averaged over maps."""
    e = np.mean([csize(quant(m, FINE)) for m in maps]) / REF[FINE]
    c = np.mean([csize(quant(ndimage.uniform_filter(m, WIN, mode="nearest"), COARSE)) for m in maps]) / REF[COARSE]
    return float(e), float(c)


@torch.no_grad()
def hidden(m, X):
    cur = X.view(X.size(0), -1); outs = [cur.cpu().numpy().reshape(-1, HW, HW)]
    for i, lin in enumerate(m.linears):
        t = lin(cur); t = m.norms[i](t) if m.norms is not None else t; z = torch.tanh(t)
        if m.gate is not None: z = m.gate(cur, z)
        cur = z; outs.append(cur.cpu().numpy().reshape(-1, HW, HW))
    return outs


cfg = load_config(MAP[RUNS[0]][0])
_, _, te, meta = load_dataset(cfg["data_path"], 500, cfg["data_name"], val_frac=cfg["val_frac"], split_seed=cfg["split_seed"],
                              data_norms=cfg["data_norms"], output_path=None, num_workers=2)
X = next(iter(te))[0][:N].to(DEV)
res = {}
for k, mech in ((0, "ca"), (1, "none")):
    per = []
    for run in RUNS:
        m = build_model(load_config(MAP[run][k]), meta); m.load_state_dict(torch.load(f"{ROOT}/paper_runs/{run}/model_{mech}.pth", map_location="cpu")); m.to(DEV).eval()
        per.append([measures(H) for H in hidden(m, X)])
    a = np.array(per)                                                   # (seeds, 11 layers, 2)
    res[mech] = dict(entropy=(a[..., 0].mean(0).tolist(), a[..., 0].std(0).tolist()), complexity=(a[..., 1].mean(0).tolist(), a[..., 1].std(0).tolist()))
noise = rng.uniform(-1, 1, (N, HW, HW)); res["ref"] = dict(noise=measures(noise))
for mech in ("ca", "none"):
    e, c = res[mech]["entropy"][0], res[mech]["complexity"][0]
    print(f"{'with LI' if mech == 'ca' else 'plain  '}  entropy by layer (input,1..10): " + " ".join(f"{v:.2f}" for v in e))
    print(f"{'':9s}complexity by layer:            " + " ".join(f"{v:.2f}" for v in c))
print(f"pure noise: entropy {res['ref']['noise'][0]:.2f}, complexity {res['ref']['noise'][1]:.2f}")
json.dump(res, open(f"{ROOT}/diagnostics/fcn_coffee_complexity.json", "w"), indent=1)

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2)); L = np.arange(11)
for mech, col, lab in (("ca", CA, "with lateral inhibition"), ("none", NO, "plain")):
    for a_, key in ((ax[0], "entropy"), (ax[1], "complexity")):
        mu, sd = map(np.array, res[mech][key]); a_.plot(L, mu, "-o", color=col, lw=2.2, label=lab); a_.fill_between(L, mu - sd, mu + sd, color=col, alpha=0.25, lw=0)
    ax[2].plot(res[mech]["entropy"][0], res[mech]["complexity"][0], "-o", color=col, lw=2.0, label=lab)
    ax[2].annotate("input", (res[mech]["entropy"][0][0], res[mech]["complexity"][0][0]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax[2].annotate("L10", (res[mech]["entropy"][0][-1], res[mech]["complexity"][0][-1]), fontsize=8, xytext=(4, -10), textcoords="offset points", color=col)
for a_, key in ((ax[0], 0), (ax[1], 1)):
    a_.axhline(res["ref"]["noise"][key], color="#888", ls=":", label="pure noise"); a_.set_xlabel("layer (0 = input)"); a_.set_xticks(L)
ax[2].scatter(*res["ref"]["noise"], color="#888", marker="x", s=60, label="pure noise")
ax[0].set_ylabel("entropy (fine-grained, compressed / noise)"); ax[1].set_ylabel("apparent complexity (coarse-grained)")
ax[2].set_xlabel("entropy"); ax[2].set_ylabel("apparent complexity"); ax[2].set_title("trajectory through depth")
ax[0].set_title("entropy"); ax[1].set_title("apparent complexity")
for a_ in ax:
    a_.grid(alpha=0.2); a_.legend(frameon=False, fontsize=8.5)
    for sp in ("top", "right"): a_.spines[sp].set_visible(False)
fig.suptitle("Coffee-automaton entropy and apparent complexity of the MNIST FCN's maps across depth (5 networks per arm)", fontsize=11)
fig.tight_layout(); fig.savefig(f"{ROOT}/diagnostics/fcn_coffee_complexity.png", dpi=140); print("WROTE diagnostics/fcn_coffee_complexity.png")
