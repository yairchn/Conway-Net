# =============================================================================================
# POLICY (STRICT): EVERY paper figure must rely ONLY on results from the CLEAN repo (Conway-Net).
# Never import/cache results computed in the messy repo (ConwayNet/experiments) into a paper figure
# or its cache (e.g. _noise_resilience.json). Each row must trace to a clean paper_runs/<run> loaded
# via MAP + build_model. If a net isn't in the clean pipeline yet, train it clean first.
# (Former violation fixed 2026-10-13: WRN/emnist_bw and WRN/emnist_color now come from clean
# paper_runs/wrn_emnist and paper_runs/wrn_emnist_color.)
# =============================================================================================
"""Centralized noise-resilience figure — all clean paper nets (FCN×4, CNN×8 = no-stem CNN with the per-channel
image gate, WRN×3), LI=CA vs none.
Per net: noise-mCA curve (mean accuracy over gaussian/shot/impulse at severities 0=clean..9) vs severity.
Multi-panel by architecture; large fonts, small panels. Saves paper_runs/_noise_resilience.json + figure."""
from conwaynet import REPO_ROOT
import json, numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from conwaynet.config import load_config
from conwaynet.models import build_model
from conwaynet.data import load_dataset
from conwaynet.robustness import noise_mca_curve
DEV = "cuda" if torch.cuda.is_available() else "cpu"; ROOT = REPO_ROOT; N = 1500
MAP = json.load(open(f"{ROOT}/paper_runs/_paper_net_configs.json")); CA, NO = "#d1495b", "#30638e"
# The 10-layer FCN (equal training budget) row was dropped from the paper figure (2026-10-21); its curves remain in the
# per-type breakdown figure (paper_code/noise_breakdown_figures.py).
GRID = [["FCN2/mnist","FCN2/fashion","FCN2/kmnist","FCN2/emnist"],
        ["CNN/mnist","CNN/fashion","CNN/kmnist","CNN/emnist"],
        ["CNN/mcolor","CNN/fcolor","CNN/kcolor","CNN/ecolor"],
        ["WRN/cifar10", "WRNN/gaussian_noise", "WRNN/shot_noise", "WRNN/impulse_noise"],
        ["GCN/mnist", "GCNN/gaussian_noise", "GCNN/shot_noise", "GCNN/impulse_noise"]]
# GCN/mnist, GCNN/<noise>: the 2-layer pixel-graph GCN on MNIST (5 seeds, configs/gcn2_mnist), averaged and per type.
# WRNN/<noise>: the WRN on CIFAR-10 (5 seeds) under ONE noise type of the same 9-level ladder (WRN/cifar10 is the
# average of the three). The official CIFAR-10-C comparison (severities 1-5) is the appendix table
# (paper_code/official_c_noise_eval.py).
RUN = {"FCN/mnist":"fcn_mnist_tanh_ns_ki_d10","FCN/fashion":"fcn_fashionmnist_tanh_ns_ki_d10",
       "FCN/kmnist":"fcn_kmnist_tanh_ns_ki_d10","FCN/emnist":"fcn_emnist_tanh_ns_ki_d10",
       "CNN/mnist":"cnn_mnist_perch","CNN/fashion":"cnn_fashionmnist_perch","CNN/kmnist":"cnn_kmnist_perch",
       "CNN/emnist":"cnn_emnist_perch","CNN/mcolor":"cnn_mnist_color_perch","CNN/fcolor":"cnn_fashionmnist_color_perch",
       "CNN/kcolor":"cnn_kmnist_color_perch","CNN/ecolor":"cnn_emnist_color_perch","WRN/cifar10":"wrn_cifar10",
       "WRN/emnist_bw":"wrn_emnist","WRN/emnist_color":"wrn_emnist_color"}

def load(run, mech):
    cfg = load_config(MAP[run][0 if mech=="ca" else 1]); rd = f"{ROOT}/paper_runs/{run}"
    tr, va, te, meta = load_dataset(cfg["data_path"], cfg["batch_size"], cfg["data_name"], val_frac=cfg["val_frac"],
        split_seed=cfg["split_seed"], data_norms=cfg["data_norms"], output_path=rd, num_workers=2)
    m = build_model(cfg, meta).to(DEV); m.load_state_dict(torch.load(f"{rd}/model_{mech}.pth", map_location=DEV)); m.eval()
    return m, meta, te

def grab(te, n):
    xs, ys = [], []
    for xb, yb in te:
        xs.append(xb); ys.append(yb)
        if sum(len(z) for z in xs) >= n: break
    return torch.cat(xs)[:n].to(DEV), torch.cat(ys)[:n].to(DEV)

import os
# Seeds: every trained seed of each panel's network (FCN seeds 42-46 = base + _s43.._s46; CNN and WRN seeds 0-4 =
# base + _s1.._s4); a seed counts once both arms' checkpoints exist. Curves are cached per run.
SEED_SUFFIX = {"FCN": ["", "_s43", "_s44", "_s45", "_s46"], "CNN": ["", "_s1", "_s2", "_s3", "_s4"], "WRN": ["", "_s1", "_s2", "_s3", "_s4"]}
CACHE = f"{ROOT}/paper_runs/_noise_resilience_runs.json"
RUNRES = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
RES = {}
for row in GRID:
    for key in row:
        if key is None or key.split("/")[0] in ("WRNN", "FCN2", "GCN", "GCNN"): continue
        runs = [RUN[key] + sfx for sfx in SEED_SUFFIX[key.split("/")[0]]
                if all(os.path.exists(f"{ROOT}/paper_runs/{RUN[key] + sfx}/model_{mech}.pth") for mech in ("ca", "none"))]
        for run in runs:
            if run in RUNRES: continue
            rec = {}
            for mech in ("ca","none"):
                m, meta, te = load(run, mech); X, Y = grab(te, N)
                mn, sd = torch.tensor(meta["mean"]), torch.tensor(meta["std"])
                rec[mech] = [float(v) for v in noise_mca_curve(m, X, Y, mn, sd, n_sev=9)]
                del m; torch.cuda.empty_cache()
            RUNRES[run] = rec; json.dump(RUNRES, open(CACHE, "w"), indent=1)
        RES[key] = {mech: list(np.mean([RUNRES[r][mech] for r in runs], 0)) for mech in ("ca", "none")}
        RES[key].update({f"{mech}_sd": list(np.std([RUNRES[r][mech] for r in runs], 0)) for mech in ("ca", "none")}, n=len(runs))
        print(f"{key}: n={len(runs)}", flush=True)
        pass
        print(f"{key}: CA mca={np.mean(RES[key]['ca'][1:]):.1f} none mca={np.mean(RES[key]['none'][1:]):.1f} Δ={np.mean(RES[key]['ca'][1:])-np.mean(RES[key]['none'][1:]):+.1f}", flush=True)

# ---- figure ----
pretty = lambda k: k.replace("FCN2/","MLP-2 matched ").replace("FCN/","MLP-10 ").replace("CNN/","CNN ").replace("WRN/","WRN ").replace("mcolor","mnist-c").replace("fcolor","fashion-c").replace("kcolor","kmnist-c").replace("ecolor","emnist-c").replace("emnist_bw","emnist").replace("emnist_color","emnist-c")
SEV = list(range(10))
from conwaynet.robustness import accuracy, _corrupt_batch
CACHE_T = f"{ROOT}/paper_runs/_noise_resilience_wrn_per_type.json"
PT = json.load(open(CACHE_T)) if os.path.exists(CACHE_T) else {}
wruns = [RUN["WRN/cifar10"] + sfx for sfx in SEED_SUFFIX["WRN"]]
for run in wruns:
    if run in PT: continue
    rec = {}
    for mech in ("ca", "none"):
        m, meta, te = load(run, mech); X, Y = grab(te, N)
        mn, sd = torch.tensor(meta["mean"]), torch.tensor(meta["std"]); clean = accuracy(m, X, Y) * 100.0
        for nm in ("gaussian_noise", "shot_noise", "impulse_noise"):
            rec[f"{nm}|{mech}"] = [clean] + [accuracy(m, _corrupt_batch(X, mn, sd, nm, lvl), Y) * 100.0 for lvl in range(9)]
        del m; torch.cuda.empty_cache()
    PT[run] = rec; json.dump(PT, open(CACHE_T, "w"), indent=1); print("per-type", run, flush=True)
for nm in ("gaussian_noise", "shot_noise", "impulse_noise"):
    k = f"WRNN/{nm}"
    RES[k] = {m: list(np.mean([PT[r][f"{nm}|{m}"] for r in wruns], 0)) for m in ("ca", "none")}
    RES[k].update({f"{m}_sd": list(np.std([PT[r][f"{nm}|{m}"] for r in wruns], 0)) for m in ("ca", "none")}, n=len(wruns))
    print(f"{k}: Δ1-5 {np.mean(RES[k]['ca'][1:6]) - np.mean(RES[k]['none'][1:6]):+.1f}  Δ1-9 {np.mean(RES[k]['ca'][1:]) - np.mean(RES[k]['none'][1:]):+.1f}")
# FCN2/*: the 2-layer FCN trained to matched clean accuracy (clean code, scripts/run_matched.py, 5 seeds), scored with the
# same protocol; read from paper_runs/_fcn_matched_noise.json (paper_code/fcn_matched_noise.py)
import re
M2 = json.load(open(f"{ROOT}/paper_runs/_fcn_matched_noise.json"))
for short, ds in (("mnist", "mnist"), ("fashion", "fashionmnist"), ("kmnist", "kmnist"), ("emnist", "emnist")):
    rs = [v for k, v in M2.items() if re.match(rf"paper_runs/fcn2_matched_{ds}_s\d+\|matched$", k)]
    RES[f"FCN2/{short}"] = {m: list(np.mean([r[m] for r in rs], 0)) for m in ("ca", "none")}
    RES[f"FCN2/{short}"].update({f"{m}_sd": list(np.std([r[m] for r in rs], 0)) for m in ("ca", "none")}, n=len(rs))
# Every curve is taken from ONE scoring run, paper_runs/_noise_per_type.json (paper_code/noise_per_type.py): an averaged
# panel is the mean of the Gaussian, shot and impulse curves of each network, then mean +- s.d. over seeds, and the WRN
# per-type panels are those same curves. So the averaged panels equal the mean of the per-type ones exactly.
PT2 = json.load(open(f"{ROOT}/paper_runs/_noise_per_type.json"))
def _set(key, recs, nm=None):
    cur = {m: [np.mean([r[f"{t}|{m}"] for t in NOISE3], 0) if nm is None else np.array(r[f"{nm}|{m}"]) for r in recs] for m in ("ca", "none")}
    RES[key] = {m: list(np.mean(cur[m], 0)) for m in ("ca", "none")}
    RES[key].update({f"{m}_sd": list(np.std(cur[m], 0)) for m in ("ca", "none")}, n=len(recs))
NOISE3 = ("gaussian_noise", "shot_noise", "impulse_noise")
for row in GRID:
    for key in row:
        fam, ds = key.split("/")
        if fam in ("FCN", "CNN", "WRN"):
            _set(key, [PT2[f"{RUN[key]}{sfx}|equal"] for sfx in SEED_SUFFIX[fam]])
        elif fam == "FCN2":
            full = {"fashion": "fashionmnist"}.get(ds, ds)
            _set(key, [PT2[f"fcn2_matched_{full}_s{k}|matched"] for k in range(5)])
        elif fam == "WRNN":
            _set(key, [PT2[f"wrn_cifar10{sfx}|equal"] for sfx in SEED_SUFFIX["WRN"]], nm=ds)
        elif fam == "GCN":
            _set(key, [PT2[f"gcn2_mnist_s{k}|equal"] for k in range(5)])
        elif fam == "GCNN":
            _set(key, [PT2[f"gcn2_mnist_s{k}|equal"] for k in range(5)], nm=ds)
        print(f"{key}: Δ1-5 {np.mean(RES[key]['ca'][1:6]) - np.mean(RES[key]['none'][1:6]):+.1f}  Δ1-9 {np.mean(RES[key]['ca'][1:]) - np.mean(RES[key]['none'][1:]):+.1f}  n={RES[key]['n']}")
fig, axes = plt.subplots(len(GRID), 4, figsize=(13, 13.6))
for r, row in enumerate(GRID):
    for c, key in enumerate(row):
        ax = axes[r][c]
        if key is None or key not in RES: ax.axis("off"); continue
        ax.axvspan(1, 5, color="#e9e4d8", zorder=0, lw=0)          # standard ImageNet-C severities 1-5
        off = False; xs = range(len(RES[key]["ca"]))
        ax.plot(xs, RES[key]["ca"], "-o", color=CA, lw=2.4, ms=4, label="with lateral inhibition")
        ax.plot(xs, RES[key]["none"], "-o", color=NO, lw=2.4, ms=4, label="plain")
        if not off and RES[key]["n"] > 1:
            for mech, col in (("ca", CA), ("none", NO)):
                mu, sdv = np.array(RES[key][mech]), np.array(RES[key][f"{mech}_sd"])
                ax.fill_between(xs, mu - sdv, mu + sdv, color=col, alpha=0.35, lw=0)
                ax.errorbar(xs, mu, yerr=sdv, fmt="none", ecolor=col, elinewidth=1.2, capsize=2.5, capthick=1.2)
        fam = key.split("/")[0]
        title = ({"WRNN": "WRN cifar10 ", "GCNN": "GCN mnist "}[fam] + key.split("/")[1].split("_")[0]) if fam in ("WRNN", "GCNN") else pretty(key).replace("GCN/", "GCN ")
        ax.set_title(title, fontsize=13.5 if fam in ("WRNN", "GCNN", "FCN2") else 15, weight="bold"); ax.set_ylim(0, 100); ax.set_xlim(0, len(xs) - 1); ax.set_xticks(xs)
        ax.tick_params(labelsize=12); ax.grid(alpha=0.2)
        d5 = np.mean(RES[key]["ca"][1:6]) - np.mean(RES[key]["none"][1:6])      # standard severities 1-5
        d9 = np.mean(RES[key]["ca"][1:]) - np.mean(RES[key]["none"][1:])        # full ladder 1-9
        ax.text(0.97, 0.97, f"$\\Delta_{{1-5}}${d5:+.1f}", transform=ax.transAxes, ha="right", va="top",
                fontsize=13, weight="bold", color=(CA if d5 >= 0 else NO))
        if not off:
            ax.text(0.97, 0.83, f"$\\Delta_{{1-9}}${d9:+.1f}", transform=ax.transAxes, ha="right", va="top",
                    fontsize=10.5, color=(CA if d9 >= 0 else NO))
        if c == 0: ax.set_ylabel("accuracy (%)", fontsize=14)
        if r == len(GRID)-1: ax.set_xlabel("noise severity", fontsize=14)
        for s in ("top","right"): ax.spines[s].set_visible(False)
        if (r, c) == (0, 0): ax.legend(fontsize=12, frameon=False, loc="lower left")
        if (r, c) == (0, 1): ax.text(3, 3, "standard\nseverities", ha="center", va="bottom", fontsize=10, color="#7a6f55")

fig.tight_layout(rect=[0,0,1,1]); out = f"{ROOT}/paper_figures/out/Noise_ressiliance.png"; fig.savefig(out, dpi=135)
print("WROTE", out)
