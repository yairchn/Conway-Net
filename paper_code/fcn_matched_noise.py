"""Paper noise protocol on the clean matched-accuracy FCN sets (configs/fcn{2,5,10}_matched, scripts/run_matched.py).

For every finished pair: the equal-budget networks (phase 1, paper_runs/<run>/model_<mech>.pth, best-val epoch) and
the matched-accuracy networks (paper_runs/<run>/matched/model_<mech>.pth). Noise: conwaynet.robustness.noise_mca_curve
(gaussian / shot / impulse, severities 1-9), first 1500 test images. Curves are cached per network in
paper_runs/_fcn_matched_noise.json, so a rerun scores only new checkpoints. Prints, per depth / dataset / condition,
the gain with - plain over severities 1-5 and 1-9 (mean +- s.d. over the seeds available).
Run: PYTHONPATH=. python paper_code/fcn_matched_noise.py
"""
from conwaynet import REPO_ROOT
import glob
import json
import os
import re

import numpy as np
import torch

from conwaynet.config import load_config
from conwaynet.data import load_dataset
from conwaynet.models import build_model
from conwaynet.robustness import noise_mca_curve

ROOT = REPO_ROOT; DEV = "cuda"; N = 1500
CACHE = f"{ROOT}/paper_runs/_fcn_matched_noise.json"
cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}

for cfg_ca in sorted(glob.glob(f"{ROOT}/configs/fcn*_matched/*_ca.json")):
    cfg_none = cfg_ca.replace("_ca.json", "_none.json")
    c = load_config(cfg_ca); run = c["output_path"]
    for cond, path in (("equal", f"{ROOT}/{run}/model_{{}}.pth"), ("matched", f"{ROOT}/{run}/matched/model_{{}}.pth")):
        key = f"{run}|{cond}"
        if key in cache or not all(os.path.exists(path.format(m)) for m in ("ca", "none")):
            continue
        _, _, te, meta = load_dataset(c["data_path"], 500, c["data_name"], val_frac=c["val_frac"], split_seed=c["split_seed"],
                                      data_norms=c["data_norms"], output_path=f"{ROOT}/{run}", num_workers=2)
        X, Y = [], []
        for xb, yb in te:
            X.append(xb); Y.append(yb)
            if sum(len(x) for x in X) >= N: break
        X, Y = torch.cat(X)[:N].to(DEV), torch.cat(Y)[:N].to(DEV)
        mn, sd = torch.tensor(meta["mean"]), torch.tensor(meta["std"])
        rec = {}
        for mech, cp in (("ca", cfg_ca), ("none", cfg_none)):
            m = build_model(load_config(cp), meta); m.load_state_dict(torch.load(path.format(mech), map_location="cpu")); m.to(DEV).eval()
            rec[mech] = [float(v) for v in noise_mca_curve(m, X, Y, mn, sd, n_sev=9)]
        cache[key] = rec; json.dump(cache, open(CACHE, "w"), indent=1)
        print(f"scored {key}", flush=True)

groups = {}
for key, rec in cache.items():
    run, cond = key.split("|")
    depth, ds, seed = re.match(r"paper_runs/fcn(\d+)_matched_(\w+?)_s(\d+)$", run).groups()
    a, b = np.array(rec["ca"]), np.array(rec["none"])
    groups.setdefault((int(depth), ds, cond), []).append((a[0], b[0], (a[1:6] - b[1:6]).mean(), (a[1:] - b[1:]).mean()))
print(f"\n{'layers':6s} {'dataset':13s} {'condition':8s} n | clean with/plain | gain sev 1-5 | gain sev 1-9")
for (depth, ds, cond), v in sorted(groups.items()):
    v = np.array(v)
    print(f"{depth:6d} {ds:13s} {cond:8s} {len(v)} | {v[:, 0].mean():5.1f}/{v[:, 1].mean():5.1f}    | "
          f"{v[:, 2].mean():+5.1f} ± {v[:, 2].std():.1f} | {v[:, 3].mean():+5.1f} ± {v[:, 3].std():.1f}")
