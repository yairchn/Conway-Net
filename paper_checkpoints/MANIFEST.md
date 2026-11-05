# Paper checkpoints (canonical set)

Every checkpoint in this folder was trained with the clean code in this repo (`conwaynet/`,
`scripts/run.py` or `conwaynet.train` called from a `paper_code/` script). Every file is a **hard link**
to the file the paper figure scripts read in `paper_runs/<run>/`, so the two cannot drift apart.
`verify.py` checks this: for every checkpoint it rebuilds the network from `config_<mech>.json`
with the clean code, loads the weights strictly, re-measures held-out test accuracy against the
recorded `test_clean_acc`, and checks the file identity (inode) against `paper_runs/`.

```bash
PYTHONPATH=. python paper_checkpoints/verify.py     # last run 2026-10-13: ALL OK (47/47)
```

Each run folder: `model_{ca,none}.pth` (with LI / without LI), `config_{ca,none}.json`,
`results_{ca,none}.json` (written by `scripts/run.py`), `data_split_indices.json` (the val carve;
test = the torchvision test split).

## fcn/ — 10-layer FCN, MNIST family, grayscale (tanh, LayerNorm, no stem, B5/S3, 0.5/0.5 norms)
`fcn_{mnist,fashionmnist,kmnist,emnist}_tanh_ns_ki_d10` (one seed, 42) — figures: activation
classifier / LFPR, depth diagnostics, saliency (Fig 8), occupancy (Fig 10), noise (Fig 11), per-digit
geometry (MNIST), appendix A.3–A.5.
`fcn_mnist_tanh_ns_ki_d100` (with LI only) — depth-100 scale-separation number in the text.

## cnn/ — CNN, no stem, per-channel image gate (Section 2.4), MNIST family grayscale + colour
`cnn_{mnist,fashionmnist,kmnist,emnist}{,_color}_perch` (seed 0) and `..._perch_s1` (seed 1) —
figures: saliency (Fig 8), per-digit geometry (Fig 9), occupancy (Fig 10), noise (Fig 11), decoy
patch (Fig 12), appendix A.6. Byte-identical to the sweep runs `paper_runs/perch_<ds>_s<seed>`.
Seven networks without LI (perch_mnist_s0/s1, perch_mnistc_s0/s1, perch_{fashionc,kmnistc,emnistc}_s0)
were reused from earlier clean runs of the same config (a network without LI does not depend on
the image-gate setting); their `results_none.json` are in the `scripts/run.py` format.
`_training_logs/` — the sweep stream logs (`### TRAIN configs/perch/...` blocks).

## wrn/ — WRN-40-2 (tanh, GroupNorm, stem conv then gates)
`wrn_cifar10` (400 epochs; seeds 1-4 in `paper_runs/wrn_cifar10_s{1..4}`), `wrn_emnist` (200 epochs) — figures: occupancy
(Fig 10), noise (Fig 11), saliency (Fig 8, CIFAR-10 / EMNIST).

## controls/ — ablations used only in the appendix (filled when their retraining completes)
- `cnn_mnist_perch_controls/` — A.6: blur λ=0.9, blur λ=0.95, DoG surround modulation on the paper CNN
  (`paper_code/cnn_controls.py`); with/without LI = `cnn/cnn_mnist_perch`.
- `fcn_mnist_controls/` — A.1–A.3: FCN controls, 3 seeds each (`paper_code/fcn_controls.py`):
  ca, none, blur λ ∈ {0.25 … 1.0}, shuffled-m, random-neighbour, shallow local-mix. Metrics:
  `paper_code/fcn_controls_metrics.py`. These replace models that earlier analysis scripts trained
  in their own loop and never saved.
- `rule_sweep_seeds/` — rule-sweep figure (10-epoch FCNs, 3 seeds per rule).
- `fcn_mnist_tanh_ns_ki_d10_r0_s{42,43,44}` / `..._r2_s{42,43,44}` — seed-agreement figure (Section 4.2): the paper MNIST FCN
  with LI and a pointwise gate (`neighbourhood_radius: 0`) / a 5x5 neighbourhood (`neighbourhood_radius: 2`); everything else
  as the paper FCN. The 3x3 comparison uses `fcn/fcn_mnist_tanh_ns_ki_d10{,_s43,_s44}`. Analysis: `paper_code/fcn_seed_agreement_scales.py`,
  figure: `paper_code/fcn_seed_agreement_paper_figure.py`.

Runs not used by the paper are not part of this repository.

## Function check
`_cleanup_check/outputs.py` records the logits of every paper network listed in `_cleanup_check/paper_runs.json`
(the paper runs, the matched-accuracy MLPs and the training snapshots: 311 networks) on one fixed input batch.
`reference.npz` holds them for the code in this repository; a code change that leaves every network's function
unchanged reproduces it bitwise.
```bash
PYTHONPATH=. python paper_checkpoints/_cleanup_check/outputs.py new.npz paper_checkpoints/_cleanup_check/paper_runs.json
PYTHONPATH=. python paper_checkpoints/_cleanup_check/outputs.py --compare paper_checkpoints/_cleanup_check/reference.npz new.npz
```
