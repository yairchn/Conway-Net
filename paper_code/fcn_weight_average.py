"""Paper (Section 4.2, ensemble coherence): the 'ensemble-mean network' -- average the weights and biases of independently trained
seeds into ONE network -- with and without LI. Inference only, existing checkpoints.
For k = 1..5 seeds: weight-averaged network vs the members, vs the output (logit) ensemble.
Clean test accuracy (full MNIST test set) and noise accuracy (gaussian/shot/impulse x 5, 2000 images).
The paper reports the FCN numbers only; the CNN rows are the contrast behind the Discussion remark.
Run: PYTHONPATH=. python paper_code/fcn_weight_average.py
"""
from conwaynet import REPO_ROOT
import json

import numpy as np
import torch

from conwaynet.config import load_config
from conwaynet.data import load_dataset
from conwaynet.eval import NOISE, evaluate_mca
from conwaynet.models import build_model

ROOT = REPO_ROOT; DEV = "cuda"
NETS = {
    "FCN": ["fcn_mnist_tanh_ns_ki_d10"] + [f"fcn_mnist_tanh_ns_ki_d10_s{k}" for k in (43, 44, 45, 46)],
    "CNN": ["cnn_mnist_perch"] + [f"cnn_mnist_perch_s{k}" for k in (1, 2, 3, 4)],
}
MAP = json.load(open(f"{ROOT}/paper_runs/_paper_net_configs.json"))


class LogitEnsemble(torch.nn.Module):
    def __init__(self, ms):
        super().__init__(); self.ms = torch.nn.ModuleList(ms)

    def forward(self, x):
        return torch.stack([m(x) for m in self.ms]).mean(0)


@torch.no_grad()
def acc(m, te):
    m.eval(); hit = n = 0
    for x, y in te:
        hit += (m(x.to(DEV)).argmax(1).cpu() == y).sum().item(); n += len(y)
    return hit / n


def noise(m, cfg, meta):
    return evaluate_mca(m, cfg["data_path"], cfg["data_name"], meta["mean"], meta["std"], DEV,
                        corruptions=NOISE, n_samples=2000, batch_size=500)["noise_mCA"]


out = {}
for arch, runs in NETS.items():
    for mech in ("ca", "none"):
        cfgs = [load_config(MAP[r][0 if mech == "ca" else 1]) for r in runs]
        _, _, te, meta = load_dataset(cfgs[0]["data_path"], 1000, cfgs[0]["data_name"], val_frac=cfgs[0]["val_frac"],
                                      split_seed=cfgs[0]["split_seed"], data_norms=cfgs[0]["data_norms"], output_path=None, num_workers=2)
        sds = [torch.load(f"{ROOT}/paper_runs/{r}/model_{mech}.pth", map_location="cpu") for r in runs]
        members = []
        for c, sd in zip(cfgs, sds):
            m = build_model(c, meta); m.load_state_dict(sd); members.append(m.to(DEV).eval())
        mem = [acc(m, te) for m in members]
        row = dict(member_clean=mem, member_noise=[noise(m, cfgs[0], meta) for m in members])
        for k in (2, 3, 4, 5):
            avg = {key: torch.stack([sd[key].float() for sd in sds[:k]]).mean(0).to(sds[0][key].dtype) for key in sds[0]}
            wa = build_model(cfgs[0], meta); wa.load_state_dict(avg); wa.to(DEV).eval()
            ens = LogitEnsemble(members[:k]).to(DEV).eval()
            row[f"wavg{k}_clean"], row[f"wavg{k}_noise"] = acc(wa, te), noise(wa, cfgs[0], meta)
            row[f"ens{k}_clean"], row[f"ens{k}_noise"] = acc(ens, te), noise(ens, cfgs[0], meta)
        out[f"{arch}|{mech}"] = row
        print(f"\n{arch} {'with' if mech == 'ca' else 'without'} LI: members clean {np.mean(mem):.4f} ± {np.std(mem):.4f}, "
              f"noise {np.mean(row['member_noise']):.4f}", flush=True)
        for k in (2, 3, 4, 5):
            print(f"  {k} seeds | weight-averaged: clean {row[f'wavg{k}_clean']:.4f} noise {row[f'wavg{k}_noise']:.4f} | "
                  f"output ensemble: clean {row[f'ens{k}_clean']:.4f} noise {row[f'ens{k}_noise']:.4f}", flush=True)
json.dump(out, open(f"{ROOT}/diagnostics/fcn_weight_average.json", "w"), indent=2)
