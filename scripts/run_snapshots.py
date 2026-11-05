"""Train one config with scripts/run.py's setup and save the model at chosen epochs (training-time snapshots).

    python scripts/run_snapshots.py <config.json> <epoch> [<epoch> ...]

Uses conwaynet.train.train unchanged; its end-of-epoch step_hook saves <output_path>/snapshots/model_<mech>_ep<E>.pth
at each listed epoch (epoch 0 = the initialisation, saved before training). num_epochs is the config's (it must cover the largest listed epoch).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from run import set_seed                                         # noqa: E402
from conwaynet.config import load_config                         # noqa: E402
from conwaynet.data import load_dataset                          # noqa: E402
from conwaynet.models import build_model                         # noqa: E402
from conwaynet.train import train                                # noqa: E402


def main():
    config = load_config(sys.argv[1]); epochs = sorted(int(e) for e in sys.argv[2:])
    assert epochs and epochs[0] >= 0 and epochs[-1] <= config["num_epochs"], (epochs, config["num_epochs"])
    set_seed(config["seed"])
    loader_gen = torch.Generator(); loader_gen.manual_seed(config["seed"])
    train_loader, val_loader, test_loader, meta = load_dataset(
        config["data_path"], config["batch_size"], config["data_name"], val_frac=config["val_frac"],
        split_seed=config["split_seed"], data_norms=config.get("data_norms"), output_path=config["output_path"],
        num_workers=config["num_workers"], augmix=config["augmix_severity"] > 0, generator=loader_gen)
    model = build_model(config, meta)
    d = os.path.join(config["output_path"], "snapshots"); os.makedirs(d, exist_ok=True)

    def hook(step, epoch, end_of_epoch, m):
        if end_of_epoch and epoch in epochs:
            torch.save(m.state_dict(), os.path.join(d, f"model_{config['mechanism']}_ep{epoch}.pth"))
            print(f"saved snapshot at epoch {epoch}", flush=True)

    if epochs[0] == 0:                                           # epoch 0 = the initialisation
        torch.save(model.state_dict(), os.path.join(d, f"model_{config['mechanism']}_ep0.pth"))
    train(model, train_loader, val_loader, {**config, "num_epochs": epochs[-1]}, meta, step_hook=hook)
    return 0


if __name__ == "__main__":
    sys.exit(main())
