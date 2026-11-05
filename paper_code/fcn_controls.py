"""Appendix A.1-A.3 controls on the PAPER FCN (10 layers, MNIST), trained and SAVED with the paper loop.

Replaces the hand-rolled training inside frame_vs_rule_*.py / dense_frontier.py /
er_rule_vs_grid_controls.py (models never saved). Each arm is the paper FCN recipe
(configs/paper/tanh/fcn_mnist_tanh_ns_ki_{ca,none}.json: Adam 1e-3 cosine, 300 epochs, bs 512, wd 5e-4)
trained with conwaynet.train (the scripts/run.py loop: best-val epoch kept) at seed s, with the gate
replaced:
  ca / none     the paper gate / no gate (seed replicates of the canonical pair, which uses seed 42)
  blur_<lam>    deep local average, same frame, no rule: y = 2((1-lam) s + lam m) - 1, s = (z+1)/2,
                m = Moore mean of the input occupancy (= (1-lam) z + lam (2m - 1))
  shuffled_m    B5/S3 on the true m, spatially permuted (fixed per seed) before the rule
  randnb        B5/S3 with each cell's 8 neighbours fixed random cells instead of the Moore ring
  localmix      shallow local average of z (grid, no rule, no input): 3x3 ring mean blended at 0.5
  sblur_<lam>   shallow blur: blur_<lam> with m read from the output occupancy (z+1)/2 instead of the input
                (= localmix at lam = 0.5)
Saves paper_runs/fcn_mnist_controls/<arm>_s<seed>/{model.pth, results.json, train.log via stdout}.
Run: PYTHONPATH=. python paper_code/fcn_controls.py <arm> <seed>
"""
from conwaynet import REPO_ROOT
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn

from conwaynet.config import load_config
from conwaynet.data import load_dataset
from conwaynet.lateral_inhibition import _rule_mask, moore_mean, parse_rule
from conwaynet.models import build_model
from conwaynet.train import evaluate, train

ROOT = REPO_ROOT
CFG = f"{ROOT}/configs/paper/tanh/fcn_mnist_tanh_ns_ki_{{}}.json"
OUT = f"{ROOT}/paper_runs/fcn_mnist_controls"
HW = 28; W = HW * HW


class DeepBlur(nn.Module):
    def __init__(self, lam):
        super().__init__(); self.lam = lam

    def forward(self, x, z):
        n = x.shape[0]
        m = moore_mean(((x + 1.0) / 2.0).view(n, 1, HW, HW)).view(n, W)
        return 2.0 * ((1.0 - self.lam) * (z + 1.0) / 2.0 + self.lam * m) - 1.0


class _RuleOnM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.b, self.s = parse_rule(cfg["rule"])
        self.g, self.d, self.ng = cfg["gamma"], cfg["delta"], cfg["normalize_gates"]

    def apply_rule(self, m, z):
        B = _rule_mask(m, self.b, self.d, self.g); S = _rule_mask(m, self.s, self.d, self.g); st = (z + 1.0) / 2.0
        c = (1.0 - st) * B + st * S
        if self.ng:
            c = c / ((1.0 - st) ** 2 + st ** 2)
        return 2.0 * c - 1.0


class ShuffledM(_RuleOnM):
    def __init__(self, cfg, seed):
        super().__init__(cfg)
        self.register_buffer("perm", torch.as_tensor(np.random.default_rng(seed).permutation(W)))

    def forward(self, x, z):
        n = x.shape[0]
        return self.apply_rule(moore_mean(((x + 1.0) / 2.0).view(n, 1, HW, HW)).view(n, W)[:, self.perm], z)


class RandomNeighbour(_RuleOnM):
    def __init__(self, cfg, seed):
        super().__init__(cfg)
        rng = np.random.default_rng(seed)
        nb = np.stack([rng.choice(np.delete(np.arange(W), i), 8, replace=False) for i in range(W)])
        self.register_buffer("nb", torch.as_tensor(nb, dtype=torch.long))

    def forward(self, x, z):
        return self.apply_rule((((x + 1.0) / 2.0).view(x.shape[0], W))[:, self.nb].mean(-1), z)


class ShallowBlur(nn.Module):
    def __init__(self, lam):
        super().__init__(); self.lam = lam

    def forward(self, x, z):
        n = z.shape[0]
        m = moore_mean(((z + 1.0) / 2.0).view(n, 1, HW, HW)).view(n, W)
        return 2.0 * ((1.0 - self.lam) * (z + 1.0) / 2.0 + self.lam * m) - 1.0


class LocalMix(nn.Module):
    def __init__(self, lam=0.5):
        super().__init__(); self.lam = lam

    def forward(self, x, z):
        n = z.shape[0]
        s = ((z + 1.0) / 2.0).view(n, 1, HW, HW)
        ring = sum(torch.roll(s, sh, dims) for sh, dims in
                   (((1,), (-1,)), ((-1,), (-1,)), ((1,), (-2,)), ((-1,), (-2,)),
                    ((1, 1), (-2, -1)), ((1, -1), (-2, -1)), ((-1, 1), (-2, -1)), ((-1, -1), (-2, -1)))) / 8.0
        return (2.0 * ((1.0 - self.lam) * s + self.lam * ring) - 1.0).view(n, W)


def build(arm, seed, meta):
    """Model for `arm` (untrained unless loaded by the caller) and its config."""
    cfg = load_config(CFG.format("none" if arm == "none" else "ca"))
    cfg["seed"] = seed
    m = build_model(cfg, meta)
    if arm.startswith("blur_"):
        m.gate = DeepBlur(float(arm.split("_")[1]))
    elif arm.startswith("sblur_"):
        m.gate = ShallowBlur(float(arm.split("_")[1]))
    elif arm == "shuffled_m":
        m.gate = ShuffledM(cfg, seed)
    elif arm == "randnb":
        m.gate = RandomNeighbour(cfg, seed)
    elif arm == "localmix":
        m.gate = LocalMix(0.5)
    elif arm not in ("ca", "none"):
        raise ValueError(arm)
    return m, cfg


def main(arm, seed):
    out = f"{OUT}/{arm}_s{seed}"
    os.makedirs(out, exist_ok=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    cfg0 = load_config(CFG.format("ca"))
    gen = torch.Generator(); gen.manual_seed(seed)
    tr, va, te, meta = load_dataset(cfg0["data_path"], cfg0["batch_size"], cfg0["data_name"], val_frac=cfg0["val_frac"],
                                    split_seed=cfg0["split_seed"], data_norms=cfg0["data_norms"], output_path=out,
                                    num_workers=cfg0["num_workers"], generator=gen)
    model, cfg = build(arm, seed, meta)
    best, last, bs = train(model, tr, va, cfg, meta, step_hook=None)
    if bs is not None:
        model.load_state_dict({k: v.to(cfg["device"]) for k, v in bs.items()})
    clean = evaluate(model, te, cfg["device"])
    torch.save(model.state_dict(), f"{out}/model.pth")
    json.dump(dict(arm=arm, seed=seed, best_val_acc=best, last_val_acc=last, test_clean_acc=clean,
                   config=CFG.format("none" if arm == "none" else "ca")), open(f"{out}/results.json", "w"), indent=2)
    print(f"{arm} s{seed}: best val {best:.4f}  test {clean:.4f}  -> {out}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
