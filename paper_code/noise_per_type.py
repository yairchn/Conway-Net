"""Per-noise-type accuracy curves (Gaussian, shot, impulse; clean + severities 1-9) for every network in the noise
figure: FCN-10 (4 datasets x 5 seeds), FCN-2 at matched accuracy (4 x 5), CNN grayscale and colour (8 x 5), WRN on
CIFAR-10 (5 seeds), the 2-layer GCN on MNIST (5 seeds, key gcn2_mnist_s<k>|equal); with and without lateral inhibition. Same protocol as paper_noise_resilience.py (first 1500 test
images, conwaynet.robustness ladders, deterministic noise), so the mean of the three types equals that figure's curve,
which this script checks. Cache: paper_runs/_noise_per_type.json, key "<run>|<cond>" -> {"<noise>|<mech>": [10 values]}.
With --uniform: the uniform-noise blend curve (robustness.noise_resilience_curve, fractions 0, 0.1, ..., 1) instead,
cached in paper_runs/_noise_uniform_blend.json.
--redo <noise>: recompute that noise type in place (used 2026-10-18 for impulse noise after its seeding fix).
Run: PYTHONPATH=. python paper_code/noise_per_type.py [--uniform | --redo impulse_noise]
"""
from conwaynet import REPO_ROOT
import glob, json, os
import numpy as np, torch
from conwaynet.config import load_config
from conwaynet.data import load_dataset
from conwaynet.models import build_model
from conwaynet.robustness import accuracy, _corrupt_batch, noise_resilience_curve, NOISE

ROOT = REPO_ROOT; DEV = "cuda"; N = 1500
MAP = json.load(open(f"{ROOT}/paper_runs/_paper_net_configs.json"))
import sys
UNIFORM = "--uniform" in sys.argv
REDO = sys.argv[sys.argv.index("--redo") + 1] if "--redo" in sys.argv else None   # recompute one noise type in place          # second pass: the uniform-noise blend (11 fractions 0..1) instead of the 3 ladders
CACHE = f"{ROOT}/paper_runs/_noise_{'uniform_blend' if UNIFORM else 'per_type'}.json"
OUT = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
BASE = {"FCN": ["fcn_mnist_tanh_ns_ki_d10", "fcn_fashionmnist_tanh_ns_ki_d10", "fcn_kmnist_tanh_ns_ki_d10", "fcn_emnist_tanh_ns_ki_d10"],
        "CNN": [f"cnn_{d}_perch" for d in ("mnist", "fashionmnist", "kmnist", "emnist", "mnist_color", "fashionmnist_color",
                                             "kmnist_color", "emnist_color")],
        "WRN": ["wrn_cifar10"]}
SFX = {"FCN": ["", "_s43", "_s44", "_s45", "_s46"], "CNN": ["", "_s1", "_s2", "_s3", "_s4"], "WRN": ["", "_s1", "_s2", "_s3", "_s4"]}
jobs = []                                   # (key, {mech: (config path, weights path)})
for arch, runs in BASE.items():
    for b in runs:
        for s in SFX[arch]:
            r = b + s
            jobs.append((f"{r}|equal", {m: (f"{ROOT}/{MAP[r][i]}", f"{ROOT}/paper_runs/{r}/model_{m}.pth") for i, m in enumerate(("ca", "none"))}))
for ds in ("mnist", "fashionmnist", "kmnist", "emnist"):
    for k in range(5):
        jobs.append((f"fcn2_matched_{ds}_s{k}|matched", {m: (f"{ROOT}/configs/fcn2_matched/{ds}_s{k}_{m}.json",
                     f"{ROOT}/paper_runs/fcn2_matched_{ds}_s{k}/matched/model_{m}.pth") for m in ("ca", "none")}))
for k in range(5):
    jobs.append((f"gcn2_mnist_s{k}|equal", {m: (f"{ROOT}/configs/gcn2_mnist/gcn2_mnist_s{k}_{m}.json",
                 f"{ROOT}/paper_runs/gcn2_mnist/s{k}/model_{m}.pth") for m in ("ca", "none")}))

for key, arms in jobs:
    if key in OUT and (REDO is None or OUT[key].get("_redone_" + REDO)): continue
    rec = dict(OUT.get(key, {})) if REDO else {}
    for mech, (cp, wp) in arms.items():
        cfg = load_config(cp)
        _, _, te, meta = load_dataset(cfg["data_path"], 500, cfg["data_name"], val_frac=cfg["val_frac"], split_seed=cfg["split_seed"],
                                      data_norms=cfg["data_norms"], output_path=None, num_workers=2)
        X, Y = [], []
        for xb, yb in te:
            X.append(xb); Y.append(yb)
            if sum(len(x) for x in X) >= N: break
        X, Y = torch.cat(X)[:N].to(DEV), torch.cat(Y)[:N].to(DEV)
        m = build_model(cfg, meta).to(DEV); m.load_state_dict(torch.load(wp, map_location=DEV)); m.eval()
        mn, sd = torch.tensor(meta["mean"]), torch.tensor(meta["std"])
        if UNIFORM:
            rec[f"uniform_blend|{mech}"] = [float(v) for v in noise_resilience_curve(m, X, Y, mn, sd)[1]]
            del m; torch.cuda.empty_cache(); continue
        clean = accuracy(m, X, Y) * 100.0
        for nm in ((REDO,) if REDO else NOISE):
            rec[f"{nm}|{mech}"] = [clean] + [accuracy(m, _corrupt_batch(X, mn, sd, nm, lvl), Y) * 100.0 for lvl in range(9)]
        del m; torch.cuda.empty_cache()
    if REDO: rec["_redone_" + REDO] = True
    OUT[key] = rec; json.dump(OUT, open(CACHE, "w"), indent=1)
    if UNIFORM:
        print(key, f"uniform Δ(t=0.1..1) {np.mean(rec['uniform_blend|ca'][1:]) - np.mean(rec['uniform_blend|none'][1:]):+.1f}", flush=True); continue
    print(key, " ".join(f"{nm.split('_')[0]} Δ1-9 {np.mean(rec[f'{nm}|ca'][1:]) - np.mean(rec[f'{nm}|none'][1:]):+.1f}" for nm in NOISE), flush=True)

if UNIFORM: sys.exit(0)
# consistency: mean over the three types = the averaged curves the figure uses
A = json.load(open(f"{ROOT}/paper_runs/_noise_resilience_runs.json")); B = json.load(open(f"{ROOT}/paper_runs/_fcn_matched_noise.json"))
worst = 0.0
for key, rec in OUT.items():
    run, cond = key.split("|"); ref = A.get(run) if cond == "equal" else B.get(f"paper_runs/{run}|matched")
    if ref is None: continue
    for mech in ("ca", "none"):
        avg = np.mean([rec[f"{nm}|{mech}"][1:] for nm in NOISE], 0)
        worst = max(worst, float(np.abs(avg - np.array(ref[mech][1:])).max()))
print(f"{len(OUT)} networks; max |mean of types - figure curve| = {worst:.4f} points")
