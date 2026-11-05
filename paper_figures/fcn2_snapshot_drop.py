"""Does the noise drop relative to clean accuracy shrink as clean accuracy grows? 2-layer clean FCN, seed 0, both arms,
snapshots at epochs 10, 25, 50, 100, 200, 500 (scripts/run_snapshots.py, paper_runs/fcn2_snapshots_<ds>/snapshots).
Per snapshot: clean accuracy and the paper noise protocol (noise_mca_curve, first 1500 test images); plotted is the mean
drop (accuracy - clean accuracy) over severities 1-5 against clean accuracy, one line per arm along training. Overlaid:
the 5-seed matched-accuracy and equal-budget networks of the 2-layer set (paper_runs/_fcn_matched_noise.json).
Run: PYTHONPATH=. python paper_figures/fcn2_snapshot_drop.py
"""
from conwaynet import REPO_ROOT
import json
import re

import numpy as np
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from conwaynet.config import load_config
from conwaynet.data import load_dataset
from conwaynet.models import build_model
from conwaynet.robustness import noise_mca_curve

ROOT = REPO_ROOT; DEV = "cuda"; N = 1500; EPOCHS = (10, 25, 50, 100, 200, 500); CA, NO = "#d1495b", "#30638e"
DS = [("mnist", "MNIST"), ("fashionmnist", "FashionMNIST"), ("kmnist", "KMNIST"), ("emnist", "EMNIST")]
res = {}
for ds, _ in DS:
    for mech in ("ca", "none"):
        cfg = load_config(f"{ROOT}/configs/fcn2_snapshots/{ds}_{mech}.json")
        _, _, te, meta = load_dataset(cfg["data_path"], 500, cfg["data_name"], val_frac=cfg["val_frac"], split_seed=cfg["split_seed"],
                                      data_norms=cfg["data_norms"], output_path=f"{ROOT}/{cfg['output_path']}", num_workers=2)
        X, Y = [], []
        for xb, yb in te:
            X.append(xb); Y.append(yb)
            if sum(len(x) for x in X) >= N: break
        X, Y = torch.cat(X)[:N].to(DEV), torch.cat(Y)[:N].to(DEV)
        mn, sd = torch.tensor(meta["mean"]), torch.tensor(meta["std"])
        for e in EPOCHS:
            m = build_model(cfg, meta)
            m.load_state_dict(torch.load(f"{ROOT}/{cfg['output_path']}/snapshots/model_{mech}_ep{e}.pth", map_location="cpu")); m.to(DEV).eval()
            c = noise_mca_curve(m, X, Y, mn, sd, n_sev=9)
            res[f"{ds}|{mech}|{e}"] = c.tolist()
            print(f"{ds:13s} {mech:4s} ep{e:4d}: clean {c[0]:5.1f}  drop sev1-5 {(c[1:6] - c[0]).mean():+6.1f}  sev1-9 {(c[1:] - c[0]).mean():+6.1f}", flush=True)
json.dump(res, open(f"{ROOT}/paper_runs/_fcn2_snapshot_noise.json", "w"), indent=1)

ref = json.load(open(f"{ROOT}/paper_runs/_fcn_matched_noise.json"))
fig, axes = plt.subplots(1, 4, figsize=(17, 4.4))
for ax, (ds, name) in zip(axes, DS):
    for mech, col, lab in (("ca", CA, "with lateral inhibition"), ("none", NO, "plain")):
        cs = np.array([res[f"{ds}|{mech}|{e}"] for e in EPOCHS])
        x, y = cs[:, 0], (cs[:, 1:6] - cs[:, :1]).mean(1)
        ax.plot(x, y, "-o", color=col, lw=2, ms=5, label=f"{lab}, training snapshots")
        for e, xi, yi in zip(EPOCHS, x, y):
            ax.annotate(str(e), (xi, yi), fontsize=7.5, color=col, xytext=(3, 3), textcoords="offset points")
        for cond, mk in (("matched", "*"), ("equal", "s")):
            pts = np.array([v[{"ca": "ca", "none": "none"}[mech]] for k, v in ref.items() if re.match(rf"paper_runs/fcn2_matched_{ds}_s\d+\|{cond}$", k)])
            if len(pts):
                ax.scatter(pts[:, 0], (pts[:, 1:6] - pts[:, :1]).mean(1), marker=mk, s=70 if mk == "*" else 30, facecolor="none" if mk == "s" else col,
                           edgecolor=col, zorder=5, label=f"{lab}, {cond} (5 seeds)")
    ax.set_title(name, fontsize=12, weight="bold"); ax.set_xlabel("clean test accuracy (%)"); ax.grid(alpha=0.2)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
axes[0].set_ylabel("mean drop, severities 1-5\n(accuracy − clean, points)")
axes[0].legend(frameon=False, fontsize=7.5, loc="lower right")
fig.suptitle("2-layer MLP: noise drop relative to clean accuracy, along training (epochs annotated)", fontsize=12.5, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.93]); out = f"{ROOT}/paper_figures/out/fcn2_snapshot_drop.png"; fig.savefig(out, dpi=140); print("WROTE", out)
