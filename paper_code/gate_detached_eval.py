"""Compare the paper networks with the same networks retrained with every gate route cut
(paper_code/gate_detached_train.py, mode detached) and retrained unchanged (mode full, the run-to-run control).
FCN-10 MNIST (seed 42) and CNN MNIST (seed 0). Per network:
  clean test accuracy; noise accuracy (conwaynet.eval.evaluate_mca: Gaussian/shot/impulse x 5 severities, 2000 images);
  saliency on/off-digit separation (first 1000 test images, as paper_code/saliency_multiseed.py);
  FCN only: activation LFPR (class-averaged last hidden maps) and classifier-weight LFPR, logit lens at layer 1,
            scale-separation index (the functions of paper_figures/fcn_rule_coherence.py), on the first 2000 test images;
  occupancy at every gate (fraction below / in the survival band / gap / birth band / above), as gate_steering_measure.
Writes diagnostics/gate_detached_eval.json. Run: PYTHONPATH=. python paper_code/gate_detached_eval.py
"""
from conwaynet import REPO_ROOT
import importlib
import json

import numpy as np
import torch

from conwaynet.config import load_config
from conwaynet.data import load_dataset
from conwaynet.eval import evaluate_mca, NOISE
from conwaynet.models import build_model
from conwaynet.train import evaluate

ROOT = REPO_ROOT; DEV = "cuda"; HW = 28
L = importlib.import_module("conwaynet.lateral_inhibition")
_s = open(f"{ROOT}/paper_code/saliency_multiseed.py").read(); exec(_s[:_s.index("\nres = {}")])             # mask_of, separation
import sys; sys.path.insert(0, f"{ROOT}/paper_code"); import scale_separation_index as SS  # used by the ssi() of the exec below
_c = open(f"{ROOT}/paper_figures/fcn_rule_coherence.py").read()
exec(_c[_c.index("EDGES = [0, 2, 3"):_c.index("def corr(")])                                              # hidden, lfpr, ssi
_m = open(f"{ROOT}/paper_code/gate_steering_measure.py").read()
exec(_m[_m.index("EDGES = [-1e9"):_m.index("_orig = L.lateral_inhibition")]); exec(_m[_m.index("def occupancy("):_m.index("\nres = {}")])

ARMS = {"FCN": ("configs/paper/tanh/fcn_mnist_tanh_ns_ki_ca.json", "paper_runs/fcn_mnist_tanh_ns_ki_d10", "fcn_mnist"),
        "CNN": ("configs/paper/cnn_mnist_perch_ca.json", "paper_runs/cnn_mnist_perch", "cnn_mnist")}
res = {}
for net, (cfgp, paper_run, name) in ARMS.items():
    cfg = load_config(f"{ROOT}/{cfgp}")
    _, _, te, meta = load_dataset(cfg["data_path"], 500, cfg["data_name"], val_frac=cfg["val_frac"], split_seed=cfg["split_seed"],
                                  data_norms=cfg["data_norms"], output_path=f"{ROOT}/{paper_run}", num_workers=2)
    XS, YS = [], []
    for xb, yb in te:
        XS.append(xb); YS.append(yb)
        if sum(len(x) for x in XS) >= 2000: break
    XS, YS = torch.cat(XS)[:2000], torch.cat(YS)[:2000]
    masks = [mask_of(im) for im in (XS[:1000] * 0.5 + 0.5).clamp(0, 1).numpy()]
    for arm, path in (("paper", f"{ROOT}/{paper_run}/model_ca.pth"), ("retrained, full", f"{ROOT}/runs/gate_detached/{name}_full/model_ca.pth"),
                      ("retrained, gate routes cut", f"{ROOT}/runs/gate_detached/{name}_detached/model_ca.pth")):
        m = build_model(cfg, meta); m.load_state_dict(torch.load(path, map_location="cpu")); m.to(DEV).eval()
        r = dict(clean=evaluate(m, te, DEV) * 100,
                 noise=evaluate_mca(m, cfg["data_path"], cfg["data_name"], meta["mean"], meta["std"], DEV, corruptions=NOISE,
                                    n_samples=cfg["mca_n_samples"], batch_size=cfg["batch_size"])["noise_mCA"] * 100,
                 sep=separation(m, XS[:1000], YS[:1000], masks),
                 occupancy=occupancy(m, XS[:1000].to(DEV)).tolist())
        if net == "FCN":
            H = hidden(m, XS.to(DEV)); fin = H[-1].cpu().numpy(); yv = YS.numpy()
            r["act_lfpr"] = lfpr(np.stack([fin[yv == k].mean(0).reshape(HW, HW) for k in range(10)]))
            r["cls_lfpr"] = lfpr(m.classifier.weight.detach().cpu().numpy().reshape(10, HW, HW))
            with torch.no_grad():
                r["lens1"] = float((m.classifier(H[0]).argmax(1).cpu() == YS).float().mean()) * 100
            r["ssi"] = ssi(H)
        res[f"{net}|{arm}"] = r
        extra = (f" | act LFPR {r['act_lfpr']:.3f} cls LFPR {r['cls_lfpr']:.3f} lens@1 {r['lens1']:5.1f} SSI {r['ssi']:5.1f}"
                 if net == "FCN" else "")
        print(f"{net} {arm:27s} clean {r['clean']:5.1f} noise {r['noise']:5.1f} sep {r['sep']:5.2f}{extra}", flush=True)
        for i, o in enumerate(r["occupancy"]):
            print(f"     gate {i + 1:2d}  " + "  ".join(f"{b} {v:4.2f}" for b, v in zip(BANDS, o)))
json.dump(res, open(f"{ROOT}/diagnostics/gate_detached_eval.json", "w"), indent=1)
print("WROTE diagnostics/gate_detached_eval.json")
