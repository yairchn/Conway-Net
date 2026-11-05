"""Leaderboard: for an image of '1', how far is each other digit from '1' in the network's own probabilities?

MNIST, all test images of '1'; FCN and CNN with and without LI, 5 seeds each. For digit d the distance is
the geometric mean over '1' images of p(1|x)/p(d|x) = exp(mean logit gap z_1 - z_d), i.e. how many times less
likely the network finds d than the correct answer (mean over seeds of the mean gap). Each column ranks the
nine other digits from closest to furthest; colour = log of the factor, one scale for all columns.
Also fits each network's temperature T on its own validation split by NLL (Guo et al. 2017, as in
paper_code/alternative_ranking.py) and stores the calibrated gaps gap/T (gap_cal_*), which the paper figure plots.
Outputs diagnostics/one_vs_digit_leaderboard.{json,png}.
Run: PYTHONPATH=. python paper_code/one_vs_digit_leaderboard.py
"""
from conwaynet import REPO_ROOT
import json

import numpy as np
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import cm, colors

from conwaynet.config import load_config
from conwaynet.data import load_dataset
from conwaynet.models import build_model

ROOT = REPO_ROOT; DEV = "cuda"
MAP = json.load(open(f"{ROOT}/paper_runs/_paper_net_configs.json"))
NETS = {"FCN": ["fcn_mnist_tanh_ns_ki_d10"] + [f"fcn_mnist_tanh_ns_ki_d10_s{k}" for k in (43, 44, 45, 46)],
        "CNN": ["cnn_mnist_perch"] + [f"cnn_mnist_perch_s{k}" for k in (1, 2, 3, 4)]}
OTHERS = [0, 2, 3, 4, 5, 6, 7, 8, 9]
COLS = [("FCN", "ca", "FCN\nwith lateral inhibition"), ("FCN", "none", "FCN\nwithout lateral inhibition"),
        ("CNN", "ca", "CNN\nwith lateral inhibition"), ("CNN", "none", "CNN\nwithout lateral inhibition")]

cfg0 = load_config(MAP[NETS["FCN"][0]][0])
_, _, te, meta = load_dataset(cfg0["data_path"], 1000, cfg0["data_name"], val_frac=cfg0["val_frac"], split_seed=cfg0["split_seed"],
                              data_norms=cfg0["data_norms"], output_path=None, num_workers=2)
X, Y = map(torch.cat, zip(*[(x, y) for x, y in te]))
X1 = X[Y == 1]



def fit_T(Z, Y):
    """temperature minimising the validation NLL (Guo et al. 2017)"""
    t = torch.zeros(1, dtype=torch.float64, requires_grad=True); opt = torch.optim.LBFGS([t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad(); loss = torch.nn.functional.cross_entropy(Z / t.exp(), Y); loss.backward(); return loss
    opt.step(closure)
    return float(t.exp())


res = {}
for arch, mech, _ in COLS:
    gaps, temps = [], []
    for run in NETS[arch]:
        m = build_model(load_config(MAP[run][0 if mech == "ca" else 1]), meta)
        m.load_state_dict(torch.load(f"{ROOT}/paper_runs/{run}/model_{mech}.pth", map_location="cpu")); m.to(DEV).eval()
        with torch.no_grad():
            Z = torch.cat([m(X1[i:i + 500].to(DEV)).cpu() for i in range(0, len(X1), 500)]).double()
        gaps.append([float((Z[:, 1] - Z[:, d]).mean()) for d in OTHERS])
        cfg = load_config(MAP[run][0 if mech == "ca" else 1])
        _, va, _, _ = load_dataset(cfg["data_path"], 1000, cfg["data_name"], val_frac=cfg["val_frac"], split_seed=cfg["split_seed"],
                                   data_norms=cfg["data_norms"], output_path=f"{ROOT}/paper_runs/{run}", num_workers=2)
        Xva, Yva = map(torch.cat, zip(*[(x, y) for x, y in va]))
        with torch.no_grad():
            Zva = torch.cat([m(Xva[i:i + 1000].to(DEV)).cpu() for i in range(0, len(Xva), 1000)]).double()
        temps.append(fit_T(Zva, Yva))
    g = np.array(gaps); T = np.array(temps); gc = g / T[:, None]
    res[f"{arch}|{mech}"] = {"gap_mean": g.mean(0).tolist(), "gap_std": g.std(0).tolist(), "gap_seeds": g.tolist(), "T_seeds": T.tolist(),
                             "gap_cal_mean": gc.mean(0).tolist(), "gap_cal_std": gc.std(0).tolist(), "gap_cal_seeds": gc.tolist()}
json.dump(dict(digits=OTHERS, n_images=int(len(X1)), **res), open(f"{ROOT}/diagnostics/one_vs_digit_leaderboard.json", "w"), indent=2)


def fmt(g):
    """factor exp(g) written as a power of ten"""
    return f"10$^{{{g / np.log(10):.1f}}}$"


cmap = plt.get_cmap("magma_r")
fig, ax = plt.subplots(figsize=(10.5, 6.6)); ax.set_xlim(0, len(COLS)); ax.set_ylim(len(OTHERS), -1.2); ax.axis("off")
for c, (arch, mech, lab) in enumerate(COLS):
    r = res[f"{arch}|{mech}"]; order = np.argsort(r["gap_mean"])
    g0 = min(r["gap_mean"])                                               # the closest digit of this network
    shade = lambda g: 0.08 + 0.7 * min((g - g0) / np.log(10) / 3.0, 1.0)  # orders of magnitude beyond it, 0..3, one scale
    ax.text(c + 0.5, -0.6, lab, ha="center", va="center", fontsize=11, weight="bold")
    for rank, i in enumerate(order):
        g = r["gap_mean"][i]; col = cmap(shade(g))
        ax.add_patch(plt.Rectangle((c + 0.04, rank + 0.06), 0.92, 0.88, color=col, lw=0))
        txt = "white" if shade(g) > 0.45 else "black"
        ax.text(c + 0.3, rank + 0.5, str(OTHERS[i]), ha="center", va="center", fontsize=17, weight="bold", color=txt)
        ax.text(c + 0.68, rank + 0.5, fmt(g), ha="center", va="center", fontsize=12, color=txt)
for rank in range(len(OTHERS)):
    ax.text(-0.04, rank + 0.5, f"{rank + 1}", ha="right", va="center", fontsize=10, color="#555")
ax.text(-0.04, -0.6, "rank", ha="right", va="center", fontsize=10, color="#555")
fig.suptitle("For an image of '1': each other digit, ranked from closest to furthest, with how many times less likely the network\n"
             "finds it than '1' (geometric mean over all 1135 test '1's, mean over 5 seeds). Colour: how much further than the network's\n"
             "closest digit, from the same (pale) to 1000 times or more (dark), on one scale for all networks.", fontsize=11)
fig.tight_layout(rect=[0.03, 0, 1, 0.93]); fig.savefig(f"{ROOT}/diagnostics/one_vs_digit_leaderboard.png", dpi=150)
for arch, mech, lab in COLS:
    r = res[f"{arch}|{mech}"]; order = np.argsort(r["gap_mean"])
    print(lab.replace("\n", " ") + ": " + "  ".join(f"{OTHERS[i]} 10^{r['gap_mean'][i]/np.log(10):.1f}" for i in order))
print("WROTE diagnostics/one_vs_digit_leaderboard.{json,png}")
