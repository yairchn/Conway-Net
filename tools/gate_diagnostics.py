"""Pre-training (or post-training) diagnostics for a network with lateral inhibition, from its config.

For every gate, on a batch of test images:
  occupancy   where the occupancy the rule reads sits relative to the rule's bands, for each count c the interval
              ((c - delta)/8, (c + delta)/8): fraction below the lowest band ("dead": both gates ~0, the unit is
              suppressed and the trunk factor S - B ~ 0), in a survival band, in a birth band, in a band of both,
              in the gap between bands ("idle": neither gate fires), and above the highest band
  gates       mean Birth and Survival value, and the fraction of cells with |S - B| < 0.05 (trunk gradient ~ 0)
  output      std of the gate output and the fraction at the tanh rails (|y| > 0.99)
For every learned weight matrix:
  gate-route share   ||g_full - g_cut|| / ||g_full||, with g_cut the gradient when every gate route is cut (the
              occupancy computed from a detached copy of the layer input; same forward pass). 0 = the layer learns
              only through the trunk; 1 = only through the gates downstream of it (it steers their occupancy)
  grad norm   ||g_full|| (vanishing / exploding gradients across depth)

Without --checkpoint the network is the config's initialisation (seeded with the config seed, as scripts/run.py).
The batch is the torchvision test split (independent of the run's train/val carve; no split file is written).
Run: PYTHONPATH=. python tools/gate_diagnostics.py <config.json> [--checkpoint model_ca.pth] [--n 512]
"""
import argparse
import importlib
import random

import numpy as np
import torch
import torch.nn.functional as F

from conwaynet.config import load_config
from conwaynet.data import load_dataset
from conwaynet.models import build_model

L = importlib.import_module("conwaynet.lateral_inhibition")   # the module (the package re-exports a function of this name)
_orig = L.lateral_inhibition


def bands(rule, delta):
    births, survivals = L.parse_rule(rule)
    iv = lambda counts: [((c - delta) / 8.0, (c + delta) / 8.0) for c in counts]
    return iv(births), iv(survivals)


def classify(m, b_iv, s_iv):
    """Fractions of m that are dead (below every band), in survival only, birth only, both, idle (gap), above."""
    inb = lambda ivs: torch.zeros_like(m, dtype=torch.bool) if not ivs else torch.stack([(m > lo) & (m < hi) for lo, hi in ivs]).any(0)
    edges = [e for iv in b_iv + s_iv for e in iv]
    lo, hi = min(edges), max(edges)
    b, s = inb(b_iv), inb(s_iv)
    dead, above = m <= lo, m >= hi
    out = dict(dead=dead, survival=s & ~b, birth=b & ~s, both=b & s, idle=~(b | s | dead | above), above=above)
    return {k: float(v.float().mean()) for k, v in out.items()}


def gate_records(model, X):
    recs = []

    def hook(mod, args, out):
        x, z = args
        if x.dim() == 2:
            x = x.view(x.shape[0], mod.channels, mod.height, mod.width); z = z.view_as(x) if z.numel() == x.numel() else z
        cfg = mod.cfg
        m = L.neighbour_mean((x + 1.0) / 2.0, cfg["neighbourhood_radius"])
        if x.shape[1] != (out.shape[1] if out.dim() == 4 else x.shape[1]):
            if cfg["channel_reduce"] == "perch_bgref":                       # the image gate's per-channel occupancy
                flat = m.flatten(2); dev = (flat - flat.median(-1, keepdim=True).values).abs()
                m = (dev / dev.amax(-1, keepdim=True).clamp_min(1e-6)).view_as(m)
            elif cfg["channel_reduce_stage"] == "input":
                m = m.mean(1, keepdim=True)
        births, survivals = L.parse_rule(cfg["rule"])
        B = L._rule_mask(m, births, cfg["delta"], cfg["gamma"]); S = L._rule_mask(m, survivals, cfg["delta"], cfg["gamma"])
        b_iv, s_iv = bands(cfg["rule"], cfg["delta"])
        recs.append(dict(occ=classify(m, b_iv, s_iv), mean_m=float(m.mean()), B=float(B.mean()), S=float(S.mean()),
                         trunk_dead=float(((S - B).abs() < 0.05).float().mean()),
                         out_std=float(out.std()), out_rail=float((out.abs() > 0.99).float().mean()),
                         grid=tuple(x.shape[-2:]), c_in=x.shape[1]))
    hs = [mod.register_forward_hook(hook) for mod in model.modules() if isinstance(mod, L.LateralInhibitionCA)]
    with torch.no_grad():
        model(X)
    for h in hs:
        h.remove()
    return recs


def weight_grads(model, X, Y, cut):
    L.lateral_inhibition = (lambda x, z, **kw: _orig(x.detach(), z, **kw)) if cut else _orig
    model.zero_grad()
    F.cross_entropy(model(X), Y).backward()
    L.lateral_inhibition = _orig
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.ndim >= 2 and p.grad is not None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config"); ap.add_argument("--checkpoint", default=None); ap.add_argument("--n", type=int, default=512)
    a = ap.parse_args()
    cfg = load_config(a.config)
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"]); torch.cuda.manual_seed_all(cfg["seed"])
    _, _, te, meta = load_dataset(cfg["data_path"], a.n, cfg["data_name"], val_frac=cfg["val_frac"], split_seed=cfg["split_seed"],
                                  data_norms=cfg.get("data_norms"), output_path=None, num_workers=2)
    X, Y = next(iter(te)); X, Y = X.to(cfg["device"]), Y.to(cfg["device"])
    model = build_model(cfg, meta)
    if a.checkpoint:
        model.load_state_dict(torch.load(a.checkpoint, map_location="cpu"))
    model.to(cfg["device"]).eval()
    print(f"{cfg['network_class']}  rule {cfg['rule']}  gamma {cfg['gamma']}  delta {cfg['delta']}  "
          f"{'trained: ' + a.checkpoint if a.checkpoint else 'at initialisation'}  ({len(X)} test images)")

    recs = gate_records(model, X)
    print(f"\n{'gate':>4} {'grid':>7} {'Cin':>4} {'mean m':>6} | {'dead':>5} {'surv':>5} {'birth':>5} {'both':>5} {'idle':>5} {'above':>5} |"
          f" {'B':>5} {'S':>5} {'S~B':>5} | {'out std':>7} {'rails':>5}")
    for i, r in enumerate(recs):
        o = r["occ"]
        print(f"{i + 1:4d} {'x'.join(map(str, r['grid'])):>7} {r['c_in']:4d} {r['mean_m']:6.2f} | {o['dead']:5.2f} {o['survival']:5.2f} "
              f"{o['birth']:5.2f} {o['both']:5.2f} {o['idle']:5.2f} {o['above']:5.2f} | {r['B']:5.2f} {r['S']:5.2f} {r['trunk_dead']:5.2f} |"
              f" {r['out_std']:7.3f} {r['out_rail']:5.2f}")
    small = [i + 1 for i, r in enumerate(recs) if min(r["grid"]) < 5]
    if small:
        print(f"warning: gates {small} act on grids smaller than 5x5 (no meaningful neighbourhood)")

    gf, gc = weight_grads(model, X, Y, False), weight_grads(model, X, Y, True)
    print(f"\n{'weight':32s} {'grad norm':>10} {'gate-route share':>17}")
    for n in gf:
        share = float((gf[n] - gc[n]).norm() / gf[n].norm().clamp_min(1e-30))
        print(f"{n:32s} {float(gf[n].norm()):10.2e} {share:17.3f}")


if __name__ == "__main__":
    main()
