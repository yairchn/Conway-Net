"""Reviewer P9: uncertainty of the noise gains Delta_{1-5} and Delta_{1-9} (with minus without lateral inhibition, mean
over Gaussian/shot/impulse and over severities 1-5 or 1-9) of every comparison in the noise figure: FCN-10 and FCN-2 at
matched accuracy (4 grayscale datasets), CNN (4 grayscale + 4 colour), WRN on CIFAR-10; 5 seeds each. Same protocol as
paper_code/noise_per_type.py (first 1500 test images, conwaynet.robustness ladders); every network of a dataset sees
the same corrupted images, and the k-th seed with lateral inhibition is paired with the k-th plain seed.
Per comparison:
  per-seed paired differences and a t-interval over the five seed pairs (training variability);
  a class-stratified bootstrap over test images, networks fixed (test-image sampling);
  a hierarchical bootstrap resampling seed pairs and, within each, images (both; the primary interval);
  noise-draw variability: draws 1-3 re-sample every corruption with a new seed (draw 0 is the paper's), on a
  representative subset (CNN MNIST, CNN colour FashionMNIST, FCN-2 MNIST, WRN CIFAR-10); the s.d. of Delta over draws.
Writes diagnostics/noise_uncertainty.json (and per-image correctness, diagnostics/noise_uncertainty_correct.npz).
Run: PYTHONPATH=. python paper_code/noise_uncertainty.py
"""
from conwaynet import REPO_ROOT
import json
import zlib

import numpy as np
import torch

from conwaynet import corruptions as C
from conwaynet.config import load_config
from conwaynet.data import load_dataset
from conwaynet.models import build_model
from conwaynet.robustness import NOISE

ROOT = REPO_ROOT; DEV = "cuda"; N = 1500; B = 2000; DRAWS = 4
MAP = json.load(open(f"{ROOT}/paper_runs/_paper_net_configs.json"))
DS = ("mnist", "fashionmnist", "kmnist", "emnist")
COMP = {}
for d in DS:
    COMP[f"FCN-10|{d}"] = [{m: (f"{ROOT}/{MAP[r][i]}", f"{ROOT}/paper_runs/{r}/model_{m}.pth") for i, m in enumerate(("ca", "none"))}
                           for r in [f"fcn_{d}_tanh_ns_ki_d10{s}" for s in ("", "_s43", "_s44", "_s45", "_s46")]]
    COMP[f"FCN-2|{d}"] = [{m: (f"{ROOT}/configs/fcn2_matched/{d}_s{k}_{m}.json", f"{ROOT}/paper_runs/fcn2_matched_{d}_s{k}/matched/model_{m}.pth")
                           for m in ("ca", "none")} for k in range(5)]
for d in DS + tuple(f"{x}_color" for x in DS):
    COMP[f"CNN|{d}"] = [{m: (f"{ROOT}/{MAP[r][i]}", f"{ROOT}/paper_runs/{r}/model_{m}.pth") for i, m in enumerate(("ca", "none"))}
                        for r in [f"cnn_{d}_perch{s}" for s in ("", "_s1", "_s2", "_s3", "_s4")]]
COMP["WRN|cifar10"] = [{m: (f"{ROOT}/{MAP[r][i]}", f"{ROOT}/paper_runs/{r}/model_{m}.pth") for i, m in enumerate(("ca", "none"))}
                       for r in [f"wrn_cifar10{s}" for s in ("", "_s1", "_s2", "_s3", "_s4")]]
COMP["GCN|mnist"] = [{m: (f"{ROOT}/configs/gcn2_mnist/gcn2_mnist_s{k}_{m}.json", f"{ROOT}/paper_runs/gcn2_mnist/s{k}/model_{m}.pth")
                      for m in ("ca", "none")} for k in range(5)]
DRAW_SUBSET = ("CNN|mnist", "CNN|fashionmnist_color", "FCN-2|mnist", "WRN|cifar10")
# --only <comparison>: compute that comparison alone and merge it into the existing outputs (added for the GCN, 2026-11-04)
import sys
if "--only" in sys.argv:
    COMP = {k: v for k, v in COMP.items() if k == sys.argv[sys.argv.index("--only") + 1]}


def corrupted_u8(x01, name, lvl, draw):
    """The paper's corruption (draw 0: conwaynet.robustness._corrupt_batch seeding) or a fresh draw; uint8 (N,H,W,3)."""
    tag = f"{name}|{lvl}" if draw == 0 else f"{name}|{lvl}|draw{draw}"
    base = np.random.RandomState(zlib.crc32(tag.encode()) % (1 << 30)); out = []
    for img in x01:
        a = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        if a.shape[2] == 1:
            a = np.repeat(a, 3, 2)
        out.append(C.apply(a, name, lvl, np.random.RandomState(base.randint(1 << 30))))
    return np.stack(out)


def to_model(u8, ch, mean, std):
    t = torch.from_numpy(u8).float().div(255).permute(0, 3, 1, 2)
    if ch == 1:
        t = t[:, :1]
    return (t - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)


@torch.no_grad()
def correct(m, X, Y):
    out = []
    for i in range(0, len(X), 500):
        o = m(X[i:i + 500].to(DEV)); o = o[0] if isinstance(o, tuple) else o
        out.append((o.argmax(1).cpu() == Y[i:i + 500]).numpy())
    return np.concatenate(out)


def deltas(Cca, Cno, w=None):
    """Cca, Cno: (seeds, noise, 9, N) bool. Per seed: Delta_{1-5}, Delta_{1-9} in points (image weights w)."""
    w = np.ones(Cca.shape[-1]) if w is None else w
    acc = lambda c: (c * w).sum(-1) / w.sum() * 100                         # (seeds, noise, 9)
    d = acc(Cca) - acc(Cno)
    return d[:, :, :5].mean((1, 2)), d.mean((1, 2))


res, store = {}, {}
if "--only" in sys.argv:
    res = json.load(open(f"{ROOT}/diagnostics/noise_uncertainty.json"))
    store = dict(np.load(f"{ROOT}/diagnostics/noise_uncertainty_correct.npz"))
for comp, pairs in COMP.items():
    cfg0 = load_config(pairs[0]["ca"][0])
    _, _, te, meta = load_dataset(cfg0["data_path"], 500, cfg0["data_name"], val_frac=cfg0["val_frac"], split_seed=cfg0["split_seed"],
                                  data_norms=cfg0["data_norms"], output_path=None, num_workers=2)
    X, Y = [], []
    for xb, yb in te:
        X.append(xb); Y.append(yb)
        if sum(len(x) for x in X) >= N: break
    X, Y = torch.cat(X)[:N], torch.cat(Y)[:N]
    mn, sd = torch.tensor(meta["mean"]), torch.tensor(meta["std"]); ch = X.shape[1]
    x01 = (X * sd.view(1, -1, 1, 1) + mn.view(1, -1, 1, 1)).clamp(0, 1)
    draws = range(DRAWS) if comp in DRAW_SUBSET else range(1)
    U = {(nm, lvl, dr): corrupted_u8(x01, nm, lvl, dr) for dr in draws for nm in NOISE for lvl in range(9)}
    Cs = {dr: {"ca": [], "none": []} for dr in draws}
    for pair in pairs:
        for mech, (cp, wp) in pair.items():
            cfg = load_config(cp)
            if cfg["data_norms"] != cfg0["data_norms"] or cfg["data_name"] != cfg0["data_name"]:
                raise ValueError(f"{cp}: data differs from the comparison's first network")
            m = build_model(cfg, meta).to(DEV); m.load_state_dict(torch.load(wp, map_location=DEV)); m.eval()
            for dr in draws:
                Cs[dr][mech].append(np.stack([np.stack([correct(m, to_model(U[(nm, lvl, dr)], ch, mn, sd), Y) for lvl in range(9)]) for nm in NOISE]))
            del m; torch.cuda.empty_cache()
    Cca, Cno = np.array(Cs[0]["ca"]), np.array(Cs[0]["none"])                 # (5, 3, 9, N)
    store[f"{comp}|ca"] = Cca; store[f"{comp}|none"] = Cno
    d5, d9 = deltas(Cca, Cno)
    t = 2.776 / np.sqrt(len(d5))                                              # t_{0.975, 4}
    rng = np.random.default_rng(0); yv = Y.numpy(); cls = [np.flatnonzero(yv == c) for c in np.unique(yv)]
    boot_img, boot_hier = [], []
    for _ in range(B):
        idx = np.concatenate([rng.choice(c, len(c)) for c in cls]); w = np.bincount(idx, minlength=N).astype(float)
        a5, a9 = deltas(Cca, Cno, w); boot_img.append((a5.mean(), a9.mean()))
        s = rng.integers(0, len(d5), len(d5)); h5, h9 = deltas(Cca[s], Cno[s], w); boot_hier.append((h5.mean(), h9.mean()))
    boot_img, boot_hier = np.array(boot_img), np.array(boot_hier)
    r = dict(per_seed_d15=d5.tolist(), per_seed_d19=d9.tolist(), d15=float(d5.mean()), d19=float(d9.mean()),
             seed_ci_d15=[float(d5.mean() - t * d5.std(ddof=1)), float(d5.mean() + t * d5.std(ddof=1))],
             seed_ci_d19=[float(d9.mean() - t * d9.std(ddof=1)), float(d9.mean() + t * d9.std(ddof=1))],
             image_ci_d15=np.percentile(boot_img[:, 0], [2.5, 97.5]).tolist(), image_ci_d19=np.percentile(boot_img[:, 1], [2.5, 97.5]).tolist(),
             hier_ci_d15=np.percentile(boot_hier[:, 0], [2.5, 97.5]).tolist(), hier_ci_d19=np.percentile(boot_hier[:, 1], [2.5, 97.5]).tolist(),
             seeds_positive_d15=int((d5 > 0).sum()), seeds_positive_d19=int((d9 > 0).sum()))
    if len(draws) > 1:
        dd = np.array([[v.mean() for v in deltas(np.array(Cs[dr]["ca"]), np.array(Cs[dr]["none"]))] for dr in draws])   # (draws, 2)
        r.update(draw_d15=dd[:, 0].tolist(), draw_d19=dd[:, 1].tolist(), draw_sd_d15=float(dd[:, 0].std(ddof=1)), draw_sd_d19=float(dd[:, 1].std(ddof=1)))
    res[comp] = r
    print(f"{comp:24s} D1-5 {r['d15']:+6.2f}  seeds [{r['seed_ci_d15'][0]:+.2f},{r['seed_ci_d15'][1]:+.2f}]  images [{r['image_ci_d15'][0]:+.2f},{r['image_ci_d15'][1]:+.2f}]  "
          f"both [{r['hier_ci_d15'][0]:+.2f},{r['hier_ci_d15'][1]:+.2f}]  pos {r['seeds_positive_d15']}/5 | D1-9 {r['d19']:+6.2f}  both [{r['hier_ci_d19'][0]:+.2f},{r['hier_ci_d19'][1]:+.2f}]"
          + (f" | draws sd {r['draw_sd_d15']:.2f}/{r['draw_sd_d19']:.2f}" if "draw_sd_d15" in r else ""), flush=True)
json.dump(res, open(f"{ROOT}/diagnostics/noise_uncertainty.json", "w"), indent=1)
np.savez_compressed(f"{ROOT}/diagnostics/noise_uncertainty_correct.npz", **store)
print("WROTE diagnostics/noise_uncertainty.json")
