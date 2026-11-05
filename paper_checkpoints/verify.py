"""Verify the canonical paper checkpoints: every checkpoint in fcn/, cnn/, wrn/ is rebuilt from its
config with the clean code (conwaynet.build_model), its weights load strictly, and its held-out test
accuracy, re-measured now, equals test_clean_acc recorded by scripts/run.py when it was trained.
Also checks that each file is the same file (hard link, same inode) as the one the paper figure
scripts read in paper_runs/<run>/.
Run: PYTHONPATH=. python paper_checkpoints/verify.py
"""
import json
import os

import torch

from conwaynet.config import load_config
from conwaynet.data import load_dataset
from conwaynet.models import build_model
from conwaynet.train import evaluate

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
bad = 0
for sub in ("fcn", "cnn", "wrn"):
    for run in sorted(os.listdir(f"{HERE}/{sub}")):
        d = f"{HERE}/{sub}/{run}"
        if not os.path.isdir(d) or run.startswith("_"):
            continue
        for mech in ("ca", "none"):
            ck = f"{d}/model_{mech}.pth"
            if not os.path.exists(ck):
                continue
            cfg = load_config(f"{d}/config_{mech}.json")
            _, _, te, meta = load_dataset(cfg["data_path"], 512, cfg["data_name"], val_frac=cfg["val_frac"],
                                          split_seed=cfg["split_seed"], data_norms=cfg.get("data_norms"),
                                          output_path=None, num_workers=4)
            m = build_model(cfg, meta)
            m.load_state_dict(torch.load(ck, map_location="cpu"), strict=True)
            acc = evaluate(m.to("cuda"), te, "cuda")
            rec = json.load(open(f"{d}/results_{mech}.json"))["test_clean_acc"]
            same = os.stat(ck).st_ino == os.stat(f"{ROOT}/paper_runs/{run}/model_{mech}.pth").st_ino
            ok = abs(acc - rec) < 5e-4 and same
            bad += not ok
            print(f"{'OK ' if ok else 'BAD'} {sub}/{run:34s} {mech:4s} test {acc:.4f} recorded {rec:.4f}  "
                  f"same file as paper_runs: {same}", flush=True)
print("ALL OK" if bad == 0 else f"{bad} PROBLEMS")
