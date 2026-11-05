"""Appendix Table A.1, 5 seeds: paper CNN on MNIST with the CA gate, plain, and the rule-free gates (blur 0.9 / 0.95,
DoG-SM). Seeds 0-4: CA and plain = paper_runs/cnn_mnist_perch{,_s1..s4}; controls = paper_runs/cnn_mnist_perch_controls
{,_s1..s4} (paper_code/cnn_controls.py). Per network: clean test accuracy, noise accuracy (conwaynet.eval.evaluate_mca:
gaussian/shot/impulse x 5 standard severities, 2000 test images) and saliency on/off separation over the first 1000
test images (as paper_code/saliency_multiseed.py). Writes diagnostics/cnn_controls_5seed.json.
Run: PYTHONPATH=. python paper_code/cnn_controls_5seed.py
The paper reports the symmetric score as region separability; the signed log-ratio is paper_code/saliency_signed_controls.py.
"""
from conwaynet import REPO_ROOT
import json
import sys

import numpy as np
import torch

sys.path.insert(0, f"{REPO_ROOT}/paper_code")
import cnn_controls as CC
_src = open(f"{REPO_ROOT}/paper_code/saliency_multiseed.py").read()
exec(_src[:_src.index("\nres = {}")])      # mask_of, separation (the module runs its loop below this point)
from conwaynet.config import load_config
from conwaynet.data import load_dataset
from conwaynet.eval import evaluate_mca, NOISE
from conwaynet.models import build_model
from conwaynet.train import evaluate

ROOT = REPO_ROOT; DEV = "cuda"
MAP = json.load(open(f"{ROOT}/paper_runs/_paper_net_configs.json")); SFX = ["", "_s1", "_s2", "_s3", "_s4"]
cfg = load_config(CC.CFG)
_, _, te, meta = load_dataset(cfg["data_path"], cfg["batch_size"], cfg["data_name"], val_frac=cfg["val_frac"], split_seed=cfg["split_seed"],
                              data_norms=cfg["data_norms"], output_path=f"{ROOT}/paper_runs/cnn_mnist_perch", num_workers=2)
X, Y = [], []
for xb, yb in te:
    X.append(xb); Y.append(yb)
    if sum(len(x) for x in X) >= 1000: break
X, Y = torch.cat(X)[:1000], torch.cat(Y)[:1000]
masks = [mask_of(im) for im in (X * 0.5 + 0.5).clamp(0, 1).numpy()]

res = {}
for arm in ("ca", "none", "blur_0.9", "blur_0.95", "dog_sm"):
    rows = []
    for sfx in SFX:
        if arm in ("ca", "none"):
            run = f"cnn_mnist_perch{sfx}"; m = build_model(load_config(MAP[run][0 if arm == "ca" else 1]), meta)
            m.load_state_dict(torch.load(f"{ROOT}/paper_runs/{run}/model_{arm}.pth", map_location="cpu"))
        else:
            m = CC.build(arm, meta, cfg); m.load_state_dict(torch.load(f"{ROOT}/paper_runs/cnn_mnist_perch_controls{sfx}/model_{arm}.pth", map_location="cpu"))
        m.to(DEV).eval()
        noise = evaluate_mca(m, cfg["data_path"], cfg["data_name"], meta["mean"], meta["std"], DEV, corruptions=NOISE,
                             n_samples=cfg["mca_n_samples"], batch_size=cfg["batch_size"])["noise_mCA"]
        rows.append(dict(clean=evaluate(m, te, DEV), noise=noise, sep=separation(m, X, Y, masks)))
    res[arm] = {k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows]))) for k in rows[0]}
    r = res[arm]
    print(f"{arm:10s} clean {r['clean'][0]*100:5.1f}±{r['clean'][1]*100:.1f}  noise {r['noise'][0]*100:5.1f}±{r['noise'][1]*100:.1f}  "
          f"sep {r['sep'][0]:5.1f}±{r['sep'][1]:.1f}", flush=True)
json.dump(res, open(f"{ROOT}/diagnostics/cnn_controls_5seed.json", "w"), indent=1)
