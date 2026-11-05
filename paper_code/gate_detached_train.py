"""Retrain a paper network with every gate route cut. Identical forward pass; in the backward pass the
neighbour mean m (and so the Birth/Survival gates) is computed from a detached copy of the layer input x, so the
layers upstream of a gate get no gradient through it and cannot learn to steer its occupancy. The trunk route
(through z, scaled by the gates) is untouched. Same config, seed, initialisation and data order as the paper network;
trained with scripts/run.py (the paper loop). mode "detached" cuts the gate routes; mode "full" retrains the same
network unchanged through the same pipeline (the run-to-run control). Writes runs/gate_detached/<name>/model_ca.pth +
results_ca.json. Run: PYTHONPATH=. python paper_code/gate_detached_train.py <paper config> <name> <detached|full>
"""
from conwaynet import REPO_ROOT
import json
import os
import sys

ROOT = REPO_ROOT
sys.path.insert(0, f"{ROOT}/scripts")
import importlib
L = importlib.import_module("conwaynet.lateral_inhibition")   # the module (the package re-exports a function of the same name)

_orig = L.lateral_inhibition


def _detached(x, z, **kw):
    return _orig(x.detach(), z, **kw)


cfg_path, name, mode = sys.argv[1], sys.argv[2], sys.argv[3]
if mode == "detached":
    L.lateral_inhibition = _detached                     # LateralInhibitionCA.forward looks the function up here
elif mode != "full":
    raise ValueError(mode)
c = json.load(open(cfg_path))
c["logging"]["output_path"] = f"runs/gate_detached/{name}"
os.makedirs(f"{ROOT}/runs/gate_detached/{name}", exist_ok=True)
tmp = f"{ROOT}/runs/gate_detached/{name}/config_ca.json"
json.dump(c, open(tmp, "w"), indent=2)
import run                                                # scripts/run.py
sys.argv = ["run.py", tmp]
run.main()
