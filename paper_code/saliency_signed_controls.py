"""Reviewer P2 for the appendix tables: the signed saliency metrics of paper_code/saliency_signed_metrics.py (log10
on/off ratio, area-normalised object mass, fraction of images favouring the object; true class, first 1000 test images,
same mask and exclusions) for
  Table blursal (MNIST CNN): input blur 0.9 / 0.95 and DoG-SM, 5 seeds (paper_runs/cnn_mnist_perch_controls{,_s1..s4},
      as paper_code/cnn_controls_5seed.py); output blur 0.9 / 0.95, seed 0 (as paper_code/deep_vs_shallow_blur.py);
  FCN blur controls (appendix text), seed 0: input blur 0.5-1.0, localmix, output blur 0.75-1.0 (paper_runs/fcn_mnist_controls);
  Table of the gate-route cut (paper_code/gate_detached_eval.py): FCN-10 and CNN on MNIST retrained unchanged ("kept") and
      with every gate route cut, one network each (runs/gate_detached/<net>_{full,detached}).
The CA and plain rows of both tables come from saliency_signed_metrics.py. Writes diagnostics/saliency_signed_controls.json.
Run: PYTHONPATH=. python paper_code/saliency_signed_controls.py
"""
from conwaynet import REPO_ROOT
import json
import sys

import numpy as np
import torch

sys.path.insert(0, f"{REPO_ROOT}/paper_code")
import cnn_controls as CC
_src = open(f"{REPO_ROOT}/paper_code/saliency_signed_metrics.py").read()
exec(_src[:_src.index("\nres, store = {}, {}")])      # load_images, evaluate, summarise (the module runs its loop below)
from conwaynet.config import load_config
from conwaynet.models import build_model

res, store = {}, {}
cfg = load_config(CC.CFG); X, Y, masks, meta = load_images(cfg, "cnn_mnist_perch")
for arm in ("blur_0.9", "blur_0.95", "dog_sm"):
    for sfx in ("", "_s1", "_s2", "_s3", "_s4"):
        m = CC.build(arm, meta, cfg); m.load_state_dict(torch.load(f"{CC.OUT}{sfx}/model_{arm}.pth", map_location="cpu")); m.to(DEV).eval()
        res[f"CNN|{arm}|cnn_mnist_perch_controls{sfx}"] = evaluate(m, X, Y, masks, f"CNN|{arm}|{sfx}", store)
for arm in ("sblur_0.9", "sblur_0.95"):
    m = CC.build(arm, meta, cfg); m.load_state_dict(torch.load(f"{CC.OUT}/model_{arm}.pth", map_location="cpu")); m.to(DEV).eval()
    res[f"CNN|{arm}|cnn_mnist_perch_controls"] = evaluate(m, X, Y, masks, f"CNN|{arm}|", store)
import fcn_controls as FC        # FCN blur controls, seed 0 (as paper_code/deep_vs_shallow_blur.py, the canonical MNIST test split)
c = load_config(FC.CFG.format("ca")); X, Y, masks, meta = load_images(c, "fcn_mnist_tanh_ns_ki_d10")
for arm in [f"blur_{l}" for l in ("0.5", "0.75", "0.9", "0.95", "1.0")] + ["localmix"] + [f"sblur_{l}" for l in ("0.75", "0.9", "0.95", "1.0")]:
    m, _ = FC.build(arm, 0, meta); m.load_state_dict(torch.load(f"{FC.OUT}/{arm}_s0/model.pth", map_location="cpu")); m.to(DEV).eval()
    res[f"FCN|{arm}|fcn_mnist_controls/{arm}_s0"] = evaluate(m, X, Y, masks, f"FCN|{arm}|", store)
for net, cfgp, run, name in (("FCN", "configs/paper/tanh/fcn_mnist_tanh_ns_ki_ca.json", "fcn_mnist_tanh_ns_ki_d10", "fcn_mnist"),
                             ("CNN", "configs/paper/cnn_mnist_perch_ca.json", "cnn_mnist_perch", "cnn_mnist")):
    cfg = load_config(f"{ROOT}/{cfgp}"); X, Y, masks, meta = load_images(cfg, run)
    for arm in ("full", "detached"):
        m = build_model(cfg, meta); m.load_state_dict(torch.load(f"{ROOT}/runs/gate_detached/{name}_{arm}/model_ca.pth", map_location="cpu")); m.to(DEV).eval()
        res[f"{net}|gate_routes_{'kept' if arm == 'full' else 'cut'}|{name}_{arm}"] = evaluate(m, X, Y, masks, f"{net}|{arm}|", store)
for k, v in res.items():
    a = v["true_class"]["all"]
    print(f"{k:50s} log10 ratio {a['log_ratio_mean']:+.2f}  obj mass {a['obj_mass_mean']:.2f}  favours obj {a['frac_favours_object']:.2f}  "
          f"region sep {a['region_sep_mean']:.1f}  excluded {a['excl_mask']}/{a['excl_zero_grad']}", flush=True)
json.dump(res, open(f"{ROOT}/diagnostics/saliency_signed_controls.json", "w"), indent=1)
print("WROTE diagnostics/saliency_signed_controls.json")
