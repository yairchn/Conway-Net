"""Matched-accuracy training (phase 2 of the noise-resilience protocol), ported from the messy repo's
scripts/train_to_accuracy.py.

    python scripts/run_matched.py <config_ca.json> <config_none.json> <target>

Phase 1 is a normal scripts/run.py run of each config with training.val_history_out set (the fixed-budget,
equal-budget networks). This script then retrains each arm from scratch, from the same seed and init:
  PART A  K = max(1, floor(crossing) - matched_safety_margin) epochs at the configured (constant) LR schedule, where
          `crossing` is the interpolated phase-1 epoch at which that arm's val accuracy first reaches `target`;
  PART B  continues from PART A with a fresh optimizer and a cosine LR decaying to matched_lr_min_factor x base over
          the remaining num_epochs - K epochs, stopping at the first epoch whose val accuracy reaches the arm's target:
          with lateral inhibition target + offset; plain target - offset, capped just below the best val accuracy the
          arm with lateral inhibition reached in PART A + PART B (so the plain network never ends above it).
The arm with lateral inhibition runs first. The model kept is the one at the stopping epoch. Saved to
<output_path>/matched/model_<mech>.pth and results_<mech>.json. `target` is a fraction (e.g. 0.965).
Config keys (training section): matched_target_offset, matched_safety_margin, matched_lr_min_factor.
"""
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from run import set_seed                                         # noqa: E402
from conwaynet.config import load_config                         # noqa: E402
from conwaynet.data import load_dataset                          # noqa: E402
from conwaynet.models import build_model                         # noqa: E402
from conwaynet.train import train, evaluate                      # noqa: E402


def crossing_epoch(curve, target):
    """Interpolated 1-based epoch at which `curve` first reaches `target` (None if never)."""
    for i, v in enumerate(curve):
        if v >= target:
            if i == 0:
                return 1.0
            lo = curve[i - 1]
            return i + (target - lo) / (v - lo)
    return None


def main():
    cfg_paths = {"ca": sys.argv[1], "none": sys.argv[2]}
    target = float(sys.argv[3])
    best_ca = None
    for mech in ("ca", "none"):
        config = load_config(cfg_paths[mech])
        assert config["mechanism"] == mech, (cfg_paths[mech], config["mechanism"])
        out = config["output_path"]
        hist = json.load(open(config["val_history_out"]))
        cross = crossing_epoch(hist, target)
        if cross is None:
            raise RuntimeError(f"{mech}: phase-1 val accuracy never reached {target} (best {max(hist):.4f})")
        K = max(1, int(math.floor(cross)) - int(config["matched_safety_margin"]))
        n_b = max(1, config["num_epochs"] - K)
        off = float(config["matched_target_offset"])
        arm_target = target + off if mech == "ca" else min(target - off, best_ca - 1e-4)
        print(f"[{mech}] phase-1 crossing {cross:.2f} -> PART A {K} epochs, PART B up to {n_b} epochs, "
              f"stop at val_acc >= {arm_target:.4f}", flush=True)

        set_seed(config["seed"])
        loader_gen = torch.Generator(); loader_gen.manual_seed(config["seed"])
        train_loader, val_loader, test_loader, meta = load_dataset(
            config["data_path"], config["batch_size"], config["data_name"],
            val_frac=config["val_frac"], split_seed=config["split_seed"], data_norms=config.get("data_norms"),
            output_path=out, num_workers=config["num_workers"], augmix=config["augmix_severity"] > 0, generator=loader_gen)
        model = build_model(config, meta)
        base = {k: v for k, v in config.items() if k != "val_history_out"}
        best_a, _, _ = train(model, train_loader, val_loader, {**base, "num_epochs": K}, meta, step_hook=None)
        best_b, last_b, _ = train(model, train_loader, val_loader,
                                  {**base, "num_epochs": n_b, "lr_schedule": "cosine",
                                   "lr_min_factor": float(config["matched_lr_min_factor"]), "stop_at_val_acc": arm_target},
                                  meta, step_hook=None)
        if mech == "ca":
            best_ca = max(best_a, best_b)
        test_clean = evaluate(model, test_loader, config["device"])
        d = os.path.join(out, "matched"); os.makedirs(d, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(d, f"model_{mech}.pth"))
        res = dict(mechanism=mech, target=target, arm_target=arm_target, phase1_crossing=cross, part_a_epochs=K,
                   part_b_max_epochs=n_b, final_val_acc=last_b, reached=bool(last_b >= arm_target),
                   test_clean_acc=test_clean, data_norms=[list(meta["mean"]), list(meta["std"])])
        json.dump(res, open(os.path.join(d, f"results_{mech}.json"), "w"), indent=2)
        print(f"[{mech}] matched: final val {last_b:.4f} (target {arm_target:.4f}), test clean {test_clean:.4f} -> {d}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
