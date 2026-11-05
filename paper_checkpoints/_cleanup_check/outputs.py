"""Logits of every paper network on one fixed input batch, to check a code change leaves every network's function
unchanged. Usage: python outputs.py <out.npz>; compare two files with python outputs.py --compare a.npz b.npz"""
from conwaynet import REPO_ROOT
import glob, json, os, sys
import numpy as np, torch

ROOT = REPO_ROOT
if sys.argv[1] == "--compare":
    a, b = np.load(sys.argv[2]), np.load(sys.argv[3]); bad = 0
    for k in a.files:
        if k not in b.files: print("MISSING", k); bad += 1; continue
        d = float(np.abs(a[k] - b[k]).max())
        if d > 0: print(f"DIFF {d:.3e} {k}"); bad += 1
    print(f"{len(a.files)} networks compared, {bad} differ / missing"); sys.exit(bool(bad))

from conwaynet.config import load_config
from conwaynet.data import DATASET_CONFIGS
from conwaynet.models import build_model

MAP = json.load(open(f"{ROOT}/paper_runs/_paper_net_configs.json"))
runs = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else []
pairs = []
for run in runs:
    for i, mech in enumerate(("ca", "none")):
        w = f"{ROOT}/paper_runs/{run}/model_{mech}.pth"
        c = f"{ROOT}/paper_runs/{run}/config_{mech}.json"
        if not os.path.exists(c) and run in MAP and len(MAP[run]) > i and MAP[run][i]: c = f"{ROOT}/{MAP[run][i]}"
        if os.path.exists(w) and os.path.exists(c): pairs.append((f"{run}|{mech}", c, w))
for c in sorted(glob.glob(f"{ROOT}/configs/fcn2_matched/*.json")):
    cfg = json.load(open(c)); out = cfg["logging"]["output_path"]; mech = cfg["lateral_inhibition"]["mechanism"]
    for w in (f"{ROOT}/{out}/model_{mech}.pth", f"{ROOT}/{out}/matched/model_{mech}.pth"):
        if os.path.exists(w): pairs.append((w.replace(ROOT + "/", ""), c, w))
for c in sorted(glob.glob(f"{ROOT}/configs/fcn2_snapshots/*.json")):
    cfg = json.load(open(c)); out = cfg["logging"]["output_path"]; mech = cfg["lateral_inhibition"]["mechanism"]
    for w in sorted(glob.glob(f"{ROOT}/{out}/snapshots/model_{mech}_ep*.pth")): pairs.append((w.replace(ROOT + "/", ""), c, w))

res = {}
for key, c, w in pairs:
    cfg = load_config(c); d = DATASET_CONFIGS[cfg["data_name"]]
    hw = 32 if cfg["data_name"] in ("CIFAR10",) else 28
    meta = dict(channels=d["channels"], num_classes=d["num_classes"], image_size=hw)
    torch.manual_seed(0); m = build_model(cfg, meta)
    m.load_state_dict(torch.load(w, map_location="cpu"), strict=True); m.eval()
    g = torch.Generator().manual_seed(1); x = torch.rand(8, d["channels"], hw, hw, generator=g) * 2 - 1
    with torch.no_grad(): res[key] = m(x).numpy()
np.savez(sys.argv[1], **res); print(len(res), "networks written to", sys.argv[1])
