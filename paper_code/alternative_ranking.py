"""Reviewer P5: is the ranking of the alternatives to the correct digit meaningful, and is it richer with lateral
inhibition, once logit scale is accounted for? log p(d|x)/p(s|x) = z_d - z_s (divided by T after temperature
scaling), so a ranking of alternatives is a ranking of logit gaps and does not depend on the scale.
MNIST FCN and CNN, with and without lateral inhibition, 5 seeds; every source digit s = 0..9, all its test images.
Per network:
  temperature T fitted on the run's validation split by NLL (Guo et al. 2017); ECE (15 bins) of the test set before
  and after; entropy and margin (top logit - second, /T for calibrated) over correctly classified test images.
Per source digit s (alternatives d != s, gap g_d = z_d - z_s per image, mean over the images of s):
  scale: calibrated log10 ratio of the top alternative and the spread (top - bottom), (mean gap)/T/ln 10;
  seed agreement: mean pairwise Spearman of the five seeds' mean-gap vectors;
  stability over images: median over images of the Spearman between the image's gap vector and the network's mean
      vector, and the share of images whose top alternative is the network's top alternative;
  resemblance: Spearman between the network's mean-gap vector and a model-free visual confusability, the share of
      the 50 nearest training images (Euclidean, raw pixels) of each test image of s that belong to d, averaged over
      the images of s; and with the pooled held-out errors of all twenty networks (counts of s -> d).
Writes diagnostics/alternative_ranking.json. Run: PYTHONPATH=. python paper_code/alternative_ranking.py
"""
from conwaynet import REPO_ROOT
import itertools
import json

import numpy as np
import torch
from scipy.stats import spearmanr

from conwaynet.config import load_config
from conwaynet.data import load_dataset
from conwaynet.models import build_model

ROOT = REPO_ROOT; DEV = "cuda"; K = 50
MAP = json.load(open(f"{ROOT}/paper_runs/_paper_net_configs.json"))
NETS = {"FCN": ["fcn_mnist_tanh_ns_ki_d10"] + [f"fcn_mnist_tanh_ns_ki_d10_s{k}" for k in (43, 44, 45, 46)],
        "CNN": ["cnn_mnist_perch"] + [f"cnn_mnist_perch_s{k}" for k in (1, 2, 3, 4)]}


def tensors(loader):
    X, Y = map(torch.cat, zip(*[(x, y) for x, y in loader]))
    return X, Y


def logits(m, X):
    with torch.no_grad():
        return torch.cat([m(X[i:i + 1000].to(DEV)).cpu() for i in range(0, len(X), 1000)]).double()


def fit_T(Z, Y):
    t = torch.zeros(1, dtype=torch.float64, requires_grad=True); opt = torch.optim.LBFGS([t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad(); loss = torch.nn.functional.cross_entropy(Z / t.exp(), Y); loss.backward(); return loss
    opt.step(closure)
    return float(t.exp())


def ece(Z, Y, bins=15):
    p = torch.softmax(Z, 1); conf, pred = p.max(1); acc = (pred == Y).double(); e = 0.0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        sel = (conf > lo) & (conf <= lo + 1 / bins)
        if sel.any():
            e += float(sel.double().mean() * (conf[sel].mean() - acc[sel].mean()).abs())
    return e


cfg0 = load_config(MAP[NETS["FCN"][0]][0])
tr, _, te, meta = load_dataset(cfg0["data_path"], 1000, cfg0["data_name"], val_frac=cfg0["val_frac"], split_seed=cfg0["split_seed"],
                               data_norms=cfg0["data_norms"], output_path=f"{ROOT}/paper_runs/{NETS['FCN'][0]}", num_workers=2)
Xte, Yte = tensors(te); Xtr, Ytr = tensors(tr)

# model-free visual confusability: share of the K nearest training images (raw pixels) in each class
A = Xtr.flatten(1).to(DEV); B = Xte.flatten(1).to(DEV); nn_lab = []
for i in range(0, len(B), 500):
    d = torch.cdist(B[i:i + 500], A); nn_lab.append(Ytr[d.topk(K, largest=False).indices.cpu()])
nn_lab = torch.cat(nn_lab)
VIS = np.zeros((10, 10))
for s in range(10):
    lab = nn_lab[Yte == s]
    VIS[s] = [(lab == d).double().mean().item() for d in range(10)]

res = {"visual_confusability_knn50": VIS.tolist()}; per_net = {}; ERR = np.zeros((10, 10))
for arch, runs in NETS.items():
    for mech in ("ca", "none"):
        nets = []
        for run in runs:
            cfg = load_config(MAP[run][0 if mech == "ca" else 1])
            _, va, _, _ = load_dataset(cfg["data_path"], 1000, cfg["data_name"], val_frac=cfg["val_frac"], split_seed=cfg["split_seed"],
                                       data_norms=cfg["data_norms"], output_path=f"{ROOT}/paper_runs/{run}", num_workers=2)
            Xva, Yva = tensors(va)
            m = build_model(cfg, meta); m.load_state_dict(torch.load(f"{ROOT}/paper_runs/{run}/model_{mech}.pth", map_location="cpu")); m.to(DEV).eval()
            Zva, Z = logits(m, Xva), logits(m, Xte)
            T = fit_T(Zva, Yva); pred = Z.argmax(1); ok = pred == Yte
            for s, d in zip(Yte[~ok].tolist(), pred[~ok].tolist()):
                ERR[s, d] += 1
            top2 = Z.topk(2, 1).values; margin = (top2[:, 0] - top2[:, 1])[ok]
            ent = lambda z: float(np.median(-(torch.softmax(z, 1) * torch.log_softmax(z, 1)).sum(1)[ok].numpy()))
            nets.append(dict(Z=Z, T=T, acc=float(ok.double().mean()), ece_raw=ece(Z, Yte), ece_cal=ece(Z / T, Yte),
                             entropy_raw_median=ent(Z), entropy_cal_median=ent(Z / T),
                             margin_raw_median=float(margin.median()), margin_cal_median=float((margin / T).median()),
                             margin_raw_q10_q90=[float(v) for v in np.percentile(margin.numpy(), [10, 90])]))
        per_net[f"{arch}|{mech}"] = nets

for key, nets in per_net.items():
    out = {k: (float(np.mean([n[k] for n in nets])), float(np.std([n[k] for n in nets])))
           for k in ("T", "acc", "ece_raw", "ece_cal", "entropy_raw_median", "entropy_cal_median", "margin_raw_median", "margin_cal_median")}
    rows = []
    for s in range(10):
        alt = [d for d in range(10) if d != s]; idx = (Yte == s)
        G = [(n["Z"][idx][:, alt] - n["Z"][idx][:, [s]]).numpy() for n in nets]           # per seed: (n_s, 9) gaps
        mean_g = np.array([g.mean(0) for g in G])                                          # (5, 9)
        seed_rho = np.mean([spearmanr(a, b)[0] for a, b in itertools.combinations(mean_g, 2)])
        img_rho = np.mean([np.median([spearmanr(gi, mg)[0] for gi in g]) for g, mg in zip(G, mean_g)])
        top_share = np.mean([np.mean(g.argmax(1) == mg.argmax()) for g, mg in zip(G, mean_g)])
        cal = np.array([mg / n["T"] / np.log(10) for mg, n in zip(mean_g, nets)])
        vis = VIS[s][alt]; err = ERR[s][alt]
        rows.append(dict(source=s, seed_spearman=float(seed_rho), image_spearman_median=float(img_rho), top_alt_share=float(top_share),
                         raw_log10_top=float((mean_g.max(1) / np.log(10)).mean()), raw_log10_spread=float(((mean_g.max(1) - mean_g.min(1)) / np.log(10)).mean()),
                         cal_log10_top=float(cal.max(1).mean()), cal_log10_spread=float((cal.max(1) - cal.min(1)).mean()),
                         rho_visual=float(np.mean([spearmanr(mg, vis)[0] for mg in mean_g])),
                         rho_errors=float(np.mean([spearmanr(mg, err)[0] for mg in mean_g])),
                         top_alt=[int(alt[i]) for i in mean_g.argmax(1)]))
    out["per_source"] = rows
    for k in ("seed_spearman", "image_spearman_median", "top_alt_share", "raw_log10_top", "raw_log10_spread", "cal_log10_top", "cal_log10_spread", "rho_visual", "rho_errors"):
        out[f"{k}_mean_over_sources"] = float(np.mean([r[k] for r in rows]))
    res[key] = out
    print(f"{key:9s} T {out['T'][0]:.2f}  ECE {out['ece_raw'][0]:.4f}->{out['ece_cal'][0]:.4f}  entropy {out['entropy_raw_median'][0]:.1e}->{out['entropy_cal_median'][0]:.1e}  "
          f"margin {out['margin_raw_median'][0]:.1f}->{out['margin_cal_median'][0]:.1f} | seeds rho {out['seed_spearman_mean_over_sources']:.2f}  images rho {out['image_spearman_median_mean_over_sources']:.2f}  "
          f"top-alt share {out['top_alt_share_mean_over_sources']:.2f} | log10 top raw {out['raw_log10_top_mean_over_sources']:.1f} cal {out['cal_log10_top_mean_over_sources']:.1f}  "
          f"spread raw {out['raw_log10_spread_mean_over_sources']:.1f} cal {out['cal_log10_spread_mean_over_sources']:.1f} | rho visual {out['rho_visual_mean_over_sources']:.2f} errors {out['rho_errors_mean_over_sources']:.2f}", flush=True)
res["pooled_errors"] = ERR.tolist()
json.dump(res, open(f"{ROOT}/diagnostics/alternative_ranking.json", "w"), indent=1)
print("WROTE diagnostics/alternative_ranking.json")
