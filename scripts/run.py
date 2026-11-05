"""Entry point: train one model from one config.

    python scripts/run.py configs/fcn_mnist.json
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch

from conwaynet import corruptions as C
from conwaynet.config import load_config
from conwaynet.data import load_dataset
from conwaynet.eval import evaluate_mca
from conwaynet.models import build_model
from conwaynet.train import train, evaluate


def set_seed(seed):
    """Seed every RNG the pipeline touches: Python, NumPy, torch (CPU+CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--init-from", default=None,
                        help="start from these weights (a saved model_<mech>.pth) instead of the configured init; "
                             "Adam state starts fresh. Recorded as init_from in results_<mech>.json")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])
    loader_gen = torch.Generator()
    loader_gen.manual_seed(config["seed"])

    train_loader, val_loader, test_loader, meta = load_dataset(
        config["data_path"], config["batch_size"], config["data_name"],
        val_frac=config["val_frac"], split_seed=config["split_seed"],
        data_norms=config.get("data_norms"), output_path=config["output_path"],
        num_workers=config["num_workers"], augmix=config["augmix_severity"] > 0,
        generator=loader_gen,
    )
    model = build_model(config, meta)
    if args.init_from is not None:
        model.load_state_dict(torch.load(args.init_from, map_location="cpu"))
        print(f"initialised from {args.init_from}", flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{config['network_class']}  mechanism={config['mechanism']}  "
          f"params={n_params:,}  dataset={config['data_name']}  device={config['device']}  "
          f"augmix={config['augmix_severity']}", flush=True)

    best, last, best_state = train(model, train_loader, val_loader, config, meta, step_hook=None)
    # Select the best-val epoch (val is carved from train, not test) and evaluate THAT model on
    # the held-out test, so the reported clean accuracy and the mCA describe the same model.
    if best_state is not None:
        model.load_state_dict({k: v.to(config["device"]) for k, v in best_state.items()})
    test_clean = evaluate(model, test_loader, config["device"])
    print(f"training done. best val_acc={best:.4f}  last-epoch val_acc={last:.4f}  "
          f"held-out test clean_acc (best-val model)={test_clean:.4f}", flush=True)

    # Save the best-val checkpoint + a structured results.json into the run folder, so the paper
    # pipeline has artifacts to analyse (interp, later mCA re-eval) — keyed by mechanism so the
    # ca and none arms of one (net, dataset) land side by side in the same output_path.
    out = config["output_path"]
    os.makedirs(out, exist_ok=True)
    mech = config["mechanism"]
    torch.save(model.state_dict(), os.path.join(out, f"model_{mech}.pth"))
    results = {"mechanism": mech, "network_class": config["network_class"],
               "data_name": config["data_name"], "activation": config["activation"],
               "n_params": n_params, "best_val_acc": best, "last_val_acc": last,
               "test_clean_acc": test_clean, "init_from": args.init_from, "data_norms": [list(meta["mean"]), list(meta["std"])]}

    if config["eval_mca"]:
        corruptions = NOISE if config["mca_corruptions"] == "noise" else C.CORRUPTIONS_15
        res = evaluate_mca(model, config["data_path"], config["data_name"], meta["mean"],
                           meta["std"], config["device"], corruptions=corruptions,
                           n_samples=config["mca_n_samples"], batch_size=config["batch_size"])
        print(f"clean={res['clean']:.4f}  mCA={res['mCA']:.4f}  noise_mCA={res['noise_mCA']:.4f}",
              flush=True)
        for name in corruptions:
            print(f"  {name:18} {[round(a, 3) for a in res[name]]}", flush=True)
        results.update(clean=res["clean"], mCA=res["mCA"], noise_mCA=res["noise_mCA"],
                       per_corruption={name: res[name] for name in corruptions})

    with open(os.path.join(out, f"results_{mech}.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved model_{mech}.pth + results_{mech}.json -> {out}", flush=True)
    return 0


NOISE = ("gaussian_noise", "shot_noise", "impulse_noise")


if __name__ == "__main__":
    sys.exit(main())
