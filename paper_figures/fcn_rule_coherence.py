"""Coherence along space, time and the ensemble, for several CA rules (MNIST FCN with LI, 5 seeds each).

Rules: B5/S3 (paper; seeds 42-46 = fcn_mnist_tanh_ns_ki_d10{,_s43..46}) and the eight other rules of the
sweep (Figure 4; B5/S5 does not train) (fcn_mnist_tanh_ns_ki_d10_<rule>_s42..46), and the network
without LI (same seeds) as the reference. All trained with the paper FCN recipe (300 epochs).
Per rule, mean +- sd over seeds (ensemble metrics over the 10 seed pairs), 1000 MNIST test images:
  space     LFPR of the class-averaged final hidden maps (act) and of the classifier weight maps (cw)
  time      logit lens at hidden layer 1 and 10 (%), scale-separation index SSI
  ensemble  unit-wise seed agreement (Pearson r of two seeds' maps, unit i vs unit i), early (1-5) and
            deep (6-10) layers, trained (TT) and untrained (UU); deep-layer agreement per spatial band;
            test accuracy of the single network obtained by averaging the 5 seeds' weights
Outputs diagnostics/fcn_rule_coherence.json (all metrics) and the figure, time and ensemble panels (space: the rule-sweep
figure, rule_sweep_300ep_figure.py).
Run: PYTHONPATH=. python paper_figures/fcn_rule_coherence.py
"""
from conwaynet import REPO_ROOT
import itertools
import json
import random

import numpy as np
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

import sys
sys.path.insert(0, f"{REPO_ROOT}/paper_code")
import scale_separation_index as SS
from conwaynet.config import load_config
from conwaynet.data import load_dataset
from conwaynet.models import build_model

ROOT = REPO_ROOT; DEV = "cuda"; HW = 28; N = 1000
SEEDS = (42, 43, 44, 45, 46)
MAP = json.load(open(f"{ROOT}/paper_runs/_paper_net_configs.json"))


def runs(tag):
    if tag in ("B5S3", "none"):
        return {k: ("fcn_mnist_tanh_ns_ki_d10" if k == 42 else f"fcn_mnist_tanh_ns_ki_d10_s{k}") for k in SEEDS}
    return {k: f"fcn_mnist_tanh_ns_ki_d10_{tag}_s{k}" for k in SEEDS}


RULES = [("B5S3", "B5/S3 (this work)", "#d1495b"), ("B5S23", "B5/S23", "#f28482"), ("B3S3", "B3/S3", "#e09f3e"), ("B3S23", "B3/S23 (Life)", "#2a9d55"),
         ("B34S34", "B34/S34 (3-4 Life)", "#b5651d"), ("B36S23", "B36/S23 (HighLife)", "#6a994e"),
         ("B3678S34678", "B3678/S34678 (Day & Night)", "#3d5a80"), ("B45678S2345", "B45678/S2345 (Walled Cities)", "#9c6644"),
         ("B2S", "B2/S (Seeds)", "#8e5ea2"), ("B1357S1357", "B1357/S1357 (Replicator)", "#30638e"),
         ("none", "without lateral inhibition", "#888")]
EDGES = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 20]
WAVE = np.array([np.sqrt(HW / EDGES[i + 1] * (HW / EDGES[i] if EDGES[i] else HW)) for i in range(len(EDGES) - 1)])
yy, xx = np.meshgrid(np.arange(HW), np.arange(HW), indexing="ij"); R = np.sqrt((yy - HW // 2) ** 2 + (xx - HW // 2) ** 2)
MASKS = [torch.tensor(((R >= EDGES[i]) & (R < EDGES[i + 1])).astype(np.float32)) for i in range(len(EDGES) - 1)]


@torch.no_grad()
def hidden(m, X):
    cur = X.view(X.size(0), -1); outs = []
    for i, lin in enumerate(m.linears):
        t = lin(cur); t = m.norms[i](t) if m.norms is not None else t; z = torch.tanh(t)
        if m.gate is not None:
            z = m.gate(cur, z)
        cur = z; outs.append(cur)
    return outs


def lfpr(maps, rf=0.25):
    x = maps - maps.mean((-2, -1), keepdims=True)
    P = np.abs(np.fft.fftshift(np.fft.fft2(x, axes=(-2, -1)), axes=(-2, -1))) ** 2
    r = R / R.max()
    return float(((P * (r <= rf)).sum((-2, -1)) / (P.sum((-2, -1)) + 1e-8)).mean())


def ssi(H):
    """Scale-separation index, the shared definition (paper_code/scale_separation_index.py)."""
    return SS.scale_separation(torch.stack([h[:SS.N_IMAGES].view(-1, HW, HW) for h in H]))["ssi"]


def corr(A, B):
    a = A.flatten(1); b = B.flatten(1); a = a - a.mean(1, keepdim=True); b = b - b.mean(1, keepdim=True)
    return float(((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-8)).mean())


def bands(M):
    F = torch.fft.fftshift(torch.fft.fft2(M), dim=(-2, -1))
    return [torch.fft.ifft2(torch.fft.ifftshift(F * k, dim=(-2, -1))).real for k in MASKS]


cfg0 = load_config(MAP[runs("B5S3")[42]][0])
_, _, te, meta = load_dataset(cfg0["data_path"], 500, cfg0["data_name"], val_frac=cfg0["val_frac"], split_seed=cfg0["split_seed"],
                              data_norms=cfg0["data_norms"], output_path=None, num_workers=2)
X, Y = [], []
for xb, yb in te:
    X.append(xb); Y.append(yb)
    if sum(len(x) for x in X) >= N: break
X, Y = torch.cat(X)[:N].to(DEV), torch.cat(Y)[:N]


@torch.no_grad()
def test_acc(m):
    hit = n = 0
    for x, y in te:
        hit += (m(x.to(DEV)).argmax(1).cpu() == y).sum().item(); n += len(y)
    return hit / n


res = {}
for tag, lab, _ in RULES:
    mech = "none" if tag == "none" else "ca"
    T, U, sds, row = {}, {}, [], {k: [] for k in ("act_lfpr", "cw_lfpr", "lens1", "lens10", "ssi", "test")}
    for sd, run in runs(tag).items():
        cfg = load_config(MAP[run][0 if mech == "ca" else 1])
        random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"]); torch.cuda.manual_seed_all(cfg["seed"])
        m = build_model(cfg, meta).to(DEV).eval(); U[sd] = [h.cpu() for h in hidden(m, X)]
        sdict = torch.load(f"{ROOT}/paper_runs/{run}/model_{mech}.pth", map_location="cpu"); sds.append(sdict)
        m.load_state_dict(sdict); m.to(DEV).eval(); H = hidden(m, X); T[sd] = [h.cpu() for h in H]
        fin = H[-1].cpu().numpy(); yv = Y.numpy()
        row["act_lfpr"].append(lfpr(np.stack([fin[yv == k].mean(0).reshape(HW, HW) for k in range(10)])))
        row["cw_lfpr"].append(lfpr(m.classifier.weight.detach().cpu().numpy().reshape(10, HW, HW)))
        with torch.no_grad():
            row["lens1"].append(float((m.classifier(H[0]).argmax(1).cpu() == Y).float().mean()) * 100)
            row["lens10"].append(float((m.classifier(H[-1]).argmax(1).cpu() == Y).float().mean()) * 100)
        row["ssi"].append(ssi(H)); row["test"].append(test_acc(m))
    L = len(T[42]); pairs = list(itertools.combinations(SEEDS, 2))
    agree = {"TT": [[corr(T[a][l], T[b][l]) for l in range(L)] for a, b in pairs],
             "UU": [[corr(U[a][l], U[b][l]) for l in range(L)] for a, b in pairs]}
    band_tt, band_uu = [], []
    for a, b in pairs:
        band_tt.append([[corr(x, y) for x, y in zip(bands(T[a][l].view(-1, HW, HW)), bands(T[b][l].view(-1, HW, HW)))] for l in range(5, L)])
        band_uu.append([[corr(x, y) for x, y in zip(bands(U[a][l].view(-1, HW, HW)), bands(U[b][l].view(-1, HW, HW)))] for l in range(5, L)])
    avg = {k: torch.stack([s[k].float() for s in sds]).mean(0).to(sds[0][k].dtype) for k in sds[0]}
    mw = build_model(load_config(MAP[runs(tag)[42]][0 if mech == "ca" else 1]), meta); mw.load_state_dict(avg); mw.to(DEV).eval()
    res[tag] = {k: (float(np.mean(v)), float(np.std(v))) for k, v in row.items()}
    for k in ("TT", "UU"):
        A = np.array(agree[k])                                    # (pairs, layers)
        res[tag][f"{k}_early"] = (float(A[:, :5].mean()), float(A[:, :5].mean(1).std()))
        res[tag][f"{k}_deep"] = (float(A[:, 5:].mean()), float(A[:, 5:].mean(1).std()))
        res[tag][f"{k}_layers"] = A.mean(0).tolist()
    res[tag]["band_TT_deep"] = np.array(band_tt).mean((0, 1)).tolist(); res[tag]["band_UU_deep"] = np.array(band_uu).mean((0, 1)).tolist()
    res[tag]["weight_avg_test"] = test_acc(mw)
    r = res[tag]
    print(f"{lab:26s} test {r['test'][0]*100:5.1f} | space: act LFPR {r['act_lfpr'][0]:.3f} cw LFPR {r['cw_lfpr'][0]:.3f} | "
          f"time: lens1 {r['lens1'][0]:5.1f} lens10 {r['lens10'][0]:5.1f} SSI {r['ssi'][0]:5.1f} | ensemble: TT early {r['TT_early'][0]:.2f} "
          f"deep {r['TT_deep'][0]:.2f} (untrained {r['UU_early'][0]:.2f}/{r['UU_deep'][0]:.2f}) | weight-avg {r['weight_avg_test']*100:5.1f}", flush=True)
    del T, U
json.dump(dict(wavelength=WAVE.tolist(), **res), open(f"{ROOT}/diagnostics/fcn_rule_coherence.json", "w"), indent=2)

fig, ax = plt.subplots(1, 3, figsize=(18.5, 5.2))   # space (activation LFPR) is panel A of the rule-sweep figure
names = [lab for _, lab, _ in RULES]; cols = [c for _, _, c in RULES]; x = np.arange(len(RULES))
ax[0].bar(x, [res[tg]["ssi"][0] for tg, _, _ in RULES], yerr=[res[tg]["ssi"][1] for tg, _, _ in RULES], color=cols, capsize=3)
ax[0].set_xticks(x); ax[0].set_xticklabels(names, rotation=35, ha="right", fontsize=8.5); ax[0].set_title("time: scale-separation index", fontsize=11)
w = 0.38
ax[1].bar(x - w / 2, [res[tg]["TT_deep"][0] for tg, _, _ in RULES], w, color=cols, label="trained")
ax[1].bar(x + w / 2, [res[tg]["UU_deep"][0] for tg, _, _ in RULES], w, color=cols, alpha=0.35, label="untrained")
ax[1].set_xticks(x); ax[1].set_xticklabels(names, rotation=35, ha="right", fontsize=8.5)
ax[1].set_title("ensemble: seed agreement, deep layers", fontsize=11); ax[1].legend(frameon=False, fontsize=9)
for tg, lab, c in RULES:
    ax[2].plot(WAVE, res[tg]["band_TT_deep"], "-o", color=c, ms=3.5, lw=2, label=lab)
ax[2].set_xscale("log"); ax[2].set_xticks([2, 3, 4, 6, 10, 20]); ax[2].set_xticklabels(["2", "3", "4", "6", "10", "20"]); ax[2].minorticks_off()
ax[2].axvline(3, color="#222", ls=":", lw=1); ax[2].set_xlabel("wavelength (px, log)")
ax[2].set_title("ensemble: deep-layer agreement by scale (trained)", fontsize=11); ax[2].legend(frameon=False, fontsize=8)
for a in ax:
    a.grid(alpha=0.2)
    for s in ("top", "right"): a.spines[s].set_visible(False)
fig.suptitle("MLP on MNIST, 5 seeds per rule: coherence in time and across the ensemble", fontsize=12)
fig.tight_layout(); fig.savefig(f"{ROOT}/paper_figures/out/fcn_rule_coherence.png", dpi=140)
print("WROTE diagnostics/fcn_rule_coherence.{json,png}")
