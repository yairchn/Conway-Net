"""Can the layers upstream of a lateral-inhibition gate steer its occupancy m, and do trained networks?

(1) Gradient share. For every learned weight matrix, the gradient of the training loss on 1000 test images, computed
    twice: normally, and with every gate route cut (m computed from a detached copy of the layer input; the forward
    pass is identical). share = ||g_full - g_cut|| / ||g_full|| is the part of the layer's update that comes from the
    gates downstream of it (0 = none); cos = cosine between the two gradients.
(2) Occupancy. At every gate site, where the occupancy the rule reads sits relative to the B5/S3 bands (gamma 15,
    delta 0.5: survival (2.5/8, 3.5/8), birth (4.5/8, 5.5/8)): fraction below the survival band, in it, in the gap,
    in the birth band, above it. The CNN's first gate reads the image (per-channel background-referenced occupancy).
Both for the trained paper networks and for the same networks at initialisation (the config seed, as scripts/run.py).
Paper FCN-10 MNIST (5 seeds) and CNN MNIST (5 seeds). Writes diagnostics/gate_steering.json.
Run: PYTHONPATH=. python paper_code/gate_steering_measure.py
"""
from conwaynet import REPO_ROOT
import json
import random

import numpy as np
import torch
import torch.nn.functional as F

import importlib
L = importlib.import_module("conwaynet.lateral_inhibition")   # the module (the package re-exports a function of the same name)
from conwaynet.config import load_config
from conwaynet.data import load_dataset
from conwaynet.models import build_model

ROOT = REPO_ROOT; DEV = "cuda"; N = 1000
MAP = json.load(open(f"{ROOT}/paper_runs/_paper_net_configs.json"))
NETS = {"FCN": ["fcn_mnist_tanh_ns_ki_d10"] + [f"fcn_mnist_tanh_ns_ki_d10_s{s}" for s in (43, 44, 45, 46)],
        "CNN": ["cnn_mnist_perch"] + [f"cnn_mnist_perch_s{s}" for s in (1, 2, 3, 4)]}
EDGES = [-1e9, 2.5 / 8, 3.5 / 8, 4.5 / 8, 5.5 / 8, 1e9]; BANDS = ("below", "survival", "gap", "birth", "above")
_orig = L.lateral_inhibition


def cut(on):
    L.lateral_inhibition = (lambda x, z, **kw: _orig(x.detach(), z, **kw)) if on else _orig


def weights(m):
    return [(n, p) for n, p in m.named_parameters() if p.ndim >= 2]


def grads(m, X, Y, cut_routes):
    cut(cut_routes); m.zero_grad()
    F.cross_entropy(m(X), Y).backward(); cut(False)
    return {n: p.grad.detach().clone() for n, p in weights(m)}


def occupancy(m, X):
    """Band fractions of the occupancy each gate reads, in call order."""
    rec = []

    def hook(mod, args):
        x = args[0]
        if x.dim() == 2:
            x = x.view(x.shape[0], mod.channels, mod.height, mod.width)
        m_ = L.neighbour_mean((x + 1.0) / 2.0, mod.cfg["neighbourhood_radius"])
        cr = mod.cfg["channel_reduce"]
        if cr == "perch_bgref" and x.shape[1] != args[1].shape[1]:                # the image gate's occupancy
            flat = m_.flatten(2); dev = (flat - flat.median(-1, keepdim=True).values).abs()
            m_ = (dev / dev.amax(-1, keepdim=True).clamp_min(1e-6)).view_as(m_)
        elif x.shape[1] != args[1].shape[1] and mod.cfg["channel_reduce_stage"] == "input":
            m_ = m_.mean(1, keepdim=True)
        h = np.histogram(m_.detach().flatten().cpu().numpy(), EDGES)[0]; rec.append(h / h.sum())
    hs = [mod.register_forward_pre_hook(hook) for mod in m.modules() if isinstance(mod, L.LateralInhibitionCA)]
    with torch.no_grad():
        m(X)
    for h in hs:
        h.remove()
    return np.array(rec)


res = {}
for net, runs in NETS.items():
    c0 = load_config(MAP[runs[0]][0])
    _, _, te, meta = load_dataset(c0["data_path"], 500, c0["data_name"], val_frac=c0["val_frac"], split_seed=c0["split_seed"],
                                  data_norms=c0["data_norms"], output_path=f"{ROOT}/paper_runs/{runs[0]}", num_workers=2)
    X, Y = [], []
    for xb, yb in te:
        X.append(xb); Y.append(yb)
        if sum(len(x) for x in X) >= N: break
    X, Y = torch.cat(X)[:N].to(DEV), torch.cat(Y)[:N].to(DEV)
    for state in ("init", "trained"):
        share, cos, occ = [], [], []
        for run in runs:
            cfg = load_config(MAP[run][0])
            random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"]); torch.cuda.manual_seed_all(cfg["seed"])
            m = build_model(cfg, meta)
            if state == "trained":
                m.load_state_dict(torch.load(f"{ROOT}/paper_runs/{run}/model_ca.pth", map_location="cpu"))
            m.to(DEV).eval()
            gf, gc = grads(m, X, Y, False), grads(m, X, Y, True)
            names = list(gf)
            share.append([float((gf[n] - gc[n]).norm() / gf[n].norm().clamp_min(1e-30)) for n in names])
            cos.append([float(F.cosine_similarity(gf[n].flatten(), gc[n].flatten(), dim=0)) for n in names])
            occ.append(occupancy(m, X))
        share, cos, occ = np.array(share), np.array(cos), np.array(occ)
        res[f"{net}|{state}"] = dict(layers=names, share=share.mean(0).tolist(), share_sd=share.std(0).tolist(),
                                     cos=cos.mean(0).tolist(), occupancy=occ.mean(0).tolist(), bands=BANDS)
        print(f"\n{net} {state} (5 networks)")
        for n, s_, sd, cs in zip(names, share.mean(0), share.std(0), cos.mean(0)):
            print(f"  grad {n:20s} gate-route share {s_:6.3f} ± {sd:.3f}  cos(full, cut) {cs:+.3f}")
        for i, o in enumerate(occ.mean(0)):
            print(f"  gate {i + 1:2d} occupancy  " + "  ".join(f"{b} {v:5.2f}" for b, v in zip(BANDS, o)))
json.dump(res, open(f"{ROOT}/diagnostics/gate_steering.json", "w"), indent=1)
print("WROTE diagnostics/gate_steering.json")
