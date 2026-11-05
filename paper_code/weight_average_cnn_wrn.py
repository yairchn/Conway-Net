"""Weight averaging of independently trained networks, the CNN and WRN counterpart
of the FCN result of Section scalesep (paper_code/fcn_rule_coherence.py: average of the seeds' state_dicts, every
parameter and buffer, loaded into one network). Paper networks, five seeds each, with and without lateral inhibition:
  CNN MNIST   paper_runs/cnn_mnist_perch{,_s1..s4}   (norm none)
  WRN CIFAR-10 paper_runs/wrn_cifar10{,_s1..s4}      (group norm: no running statistics, so a plain average is complete)
Reports the held-out test accuracy of each seed and of the networks averaged over the first k = 2..5 seeds.
Writes diagnostics/weight_average_cnn_wrn.json.
Run: PYTHONPATH=. python paper_code/weight_average_cnn_wrn.py
"""
from conwaynet import REPO_ROOT
import json

import numpy as np
import torch

from conwaynet.config import load_config
from conwaynet.data import load_dataset
from conwaynet.models import build_model

ROOT = REPO_ROOT; DEV = "cuda"
MAP = json.load(open(f"{ROOT}/paper_runs/_paper_net_configs.json"))
NETS = {"CNN MNIST": [f"cnn_mnist_perch{s}" for s in ("", "_s1", "_s2", "_s3", "_s4")],
        "WRN CIFAR-10": [f"wrn_cifar10{s}" for s in ("", "_s1", "_s2", "_s3", "_s4")]}


@torch.no_grad()
def test_acc(m, te):
    hit = n = 0
    for x, y in te:
        hit += (m(x.to(DEV)).argmax(1).cpu() == y).sum().item(); n += len(y)
    return hit / n


res = {}
for net, runs in NETS.items():
    c0 = load_config(MAP[runs[0]][0])
    _, _, te, meta = load_dataset(c0["data_path"], 500, c0["data_name"], val_frac=c0["val_frac"], split_seed=c0["split_seed"],
                                  data_norms=c0["data_norms"], output_path=f"{ROOT}/paper_runs/{runs[0]}", num_workers=4)
    for k, mech in ((0, "ca"), (1, "none")):
        sds, single = [], []
        for run in runs:
            m = build_model(load_config(MAP[run][k]), meta)
            sd = torch.load(f"{ROOT}/paper_runs/{run}/model_{mech}.pth", map_location="cpu"); sds.append(sd)
            m.load_state_dict(sd); m.to(DEV).eval(); single.append(test_acc(m, te))
        avg = {}
        for n in range(2, len(sds) + 1):
            sd = {key: torch.stack([s[key].float() for s in sds[:n]]).mean(0).to(sds[0][key].dtype) for key in sds[0]}
            m = build_model(load_config(MAP[runs[0]][k]), meta); m.load_state_dict(sd); m.to(DEV).eval()
            avg[n] = test_acc(m, te)
        res[f"{net} {mech}"] = dict(single=single, avg=avg)
        print(f"{net:13s} {mech:4s} seeds {np.mean(single)*100:5.1f}±{np.std(single)*100:.1f} | averaged over k seeds: "
              + "  ".join(f"k={n} {a*100:5.1f}" for n, a in avg.items()), flush=True)
json.dump(res, open(f"{ROOT}/diagnostics/weight_average_cnn_wrn.json", "w"), indent=1)
print("WROTE diagnostics/weight_average_cnn_wrn.json")
