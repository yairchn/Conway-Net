"""Blur control for the percolation-over-training analysis: the paper FCN / CNN with the gate replaced by the deep
blur of the controls appendix (FCN: paper_code/fcn_controls.py DeepBlur; CNN: paper_code/cnn_controls.py BlurGate,
y = (1-lam) z + lam (2 m(x) - 1), the gate's input->output wiring without the birth/survival rule), trained with
the with-LI config of the run (configs/percol_epochs/<run>_ca.json: same seed, optimiser, schedule) and saved at
every epoch, as scripts/run_snapshots.py does.
Writes runs/percol_epochs/<run>/snapshots/model_blur_<lam>_ep<E>.pth.
Run: PYTHONPATH=. python paper_code/percolation_blur_snapshots.py <run> <lam>
"""
from conwaynet import REPO_ROOT
import os
import sys

import torch

sys.path.insert(0, f"{REPO_ROOT}/paper_code"); sys.path.insert(0, f"{REPO_ROOT}/scripts")
import cnn_controls as CC                                        # noqa: E402
import fcn_controls as FC                                        # noqa: E402
from run import set_seed                                         # noqa: E402
from conwaynet.config import load_config                         # noqa: E402
from conwaynet.data import load_dataset                          # noqa: E402
from conwaynet.models import build_model                         # noqa: E402
from conwaynet.train import train                                # noqa: E402

ROOT = REPO_ROOT


def build_blur(cfg, meta, lam):
    m = build_model(cfg, meta)
    if hasattr(m, "linears"):
        m.gate = FC.DeepBlur(lam)
    else:
        m.li1, m.li2 = CC.BlurGate(lam, image_site=True), CC.BlurGate(lam, image_site=False)
    return m


def main(run, lam):
    config = load_config(f"{ROOT}/configs/percol_epochs/{run}_ca.json"); arm = f"blur_{lam:g}"
    set_seed(config["seed"])
    gen = torch.Generator(); gen.manual_seed(config["seed"])
    tr, va, _, meta = load_dataset(config["data_path"], config["batch_size"], config["data_name"], val_frac=config["val_frac"],
                                   split_seed=config["split_seed"], data_norms=config["data_norms"], output_path=config["output_path"],
                                   num_workers=config["num_workers"], augmix=config["augmix_severity"] > 0, generator=gen)
    model = build_blur(config, meta, lam)
    d = os.path.join(config["output_path"], "snapshots"); os.makedirs(d, exist_ok=True)

    def hook(step, epoch, end_of_epoch, m):
        if end_of_epoch:
            torch.save(m.state_dict(), os.path.join(d, f"model_{arm}_ep{epoch}.pth"))

    torch.save(model.state_dict(), os.path.join(d, f"model_{arm}_ep0.pth"))
    train(model, tr, va, config, meta, step_hook=hook)


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]))
