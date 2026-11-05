"""Inference-only noise robustness on the official benchmark releases.
(1) WRN-40-2 CIFAR-10 (paper checkpoints, 5 seeds, with/without LI) on CIFAR-10-C (Hendrycks & Dietterich 2019, zenodo 2535967):
    gaussian/shot/impulse/speckle noise, severities 1-5, all 10000 test images each; plus the clean test set.
(2) MNIST FCN and CNN (paper checkpoints, 5 seeds each, with/without LI) on MNIST-C (Mu & Gilmer 2019, zenodo 3239543):
    identity, shot_noise, impulse_noise (one fixed severity each), all 10000 test images.
(3) Where MNIST-C noise sits on our ladder: the mean |change| per pixel of MNIST-C shot/impulse vs identity, against the same
    measure for our protocol (conwaynet.robustness._corrupt_batch) at every ladder level.
Run: PYTHONPATH=. python paper_code/official_c_noise_eval.py
"""
from conwaynet import REPO_ROOT
import json

import numpy as np
import torch

from conwaynet.config import load_config
from conwaynet.data import load_dataset
from conwaynet.models import build_model
from conwaynet.robustness import _corrupt_batch

ROOT = REPO_ROOT; DEV = "cuda"; OFF = "/data/official_c"
MAP = json.load(open(f"{ROOT}/paper_runs/_paper_net_configs.json"))
out = {}


def load(run, k):
    cfg = load_config(MAP[run][k])
    _, _, te, meta = load_dataset(cfg["data_path"], 500, cfg["data_name"], val_frac=cfg["val_frac"], split_seed=cfg["split_seed"],
                                  data_norms=cfg["data_norms"], output_path=f"{ROOT}/paper_runs/{run}", num_workers=2)
    m = build_model(cfg, meta); mech = cfg["mechanism"]
    m.load_state_dict(torch.load(f"{ROOT}/paper_runs/{run}/model_{mech}.pth", map_location="cpu"))
    return m.to(DEV).eval(), meta, te


@torch.no_grad()
def acc_u8(m, imgs, labels, mean, std):
    """imgs uint8 (N,H,W,C); normalise as in training."""
    mn = torch.tensor(mean).view(1, -1, 1, 1).to(DEV); sd = torch.tensor(std).view(1, -1, 1, 1).to(DEV); c = 0
    for i in range(0, len(imgs), 1000):
        x = torch.from_numpy(imgs[i:i + 1000]).to(DEV).permute(0, 3, 1, 2).float() / 255
        c += (m((x - mn) / sd).argmax(1).cpu() == torch.from_numpy(labels[i:i + 1000]).long()).sum().item()
    return 100.0 * c / len(imgs)


# ---- (1) CIFAR-10-C
lab = np.load(f"{OFF}/CIFAR-10-C/labels.npy")
NO = ("gaussian_noise", "shot_noise", "impulse_noise", "speckle_noise")
WRN_RUNS = ["wrn_cifar10"] + [f"wrn_cifar10_s{k}" for k in range(1, 5)]          # 5 seeds
ARR = {c: np.load(f"{OFF}/CIFAR-10-C/{c}.npy") for c in NO}
for k, name in ((0, "with LI"), (1, "without LI")):
    per = []
    for run in WRN_RUNS:
        m, meta, te = load(run, k)
        X = torch.cat([x for x, _ in te]); Y = torch.cat([y for _, y in te])
        with torch.no_grad():
            clean = 100.0 * sum((m(X[i:i + 1000].to(DEV)).argmax(1).cpu() == Y[i:i + 1000]).sum().item() for i in range(0, len(X), 1000)) / len(X)
        r = {"clean": clean}
        for c in NO:
            r[c] = [acc_u8(m, ARR[c][s * 10000:(s + 1) * 10000], lab[s * 10000:(s + 1) * 10000], meta["mean"], meta["std"]) for s in range(5)]
        r["noise3_mean"] = float(np.mean([np.mean(r[c]) for c in NO[:3]])); out[f"WRN_CIFAR10C|{name}|{run}"] = r; per.append(r)
        del m; torch.cuda.empty_cache()
        print(f"WRN CIFAR-10-C {name} {run}: clean {clean:.2f} | " + " | ".join(f"{c} {np.mean(r[c]):.2f}" for c in NO), flush=True)
    agg = {"clean": float(np.mean([r["clean"] for r in per])), "clean_sd": float(np.std([r["clean"] for r in per])), "n": len(per)}
    for c in NO + ("noise3_mean",):
        agg[c] = list(np.mean([r[c] for r in per], 0)) if c != "noise3_mean" else float(np.mean([r[c] for r in per]))
        agg[c + "_sd"] = list(np.std([r[c] for r in per], 0)) if c != "noise3_mean" else float(np.std([r[c] for r in per]))
    out[f"WRN_CIFAR10C|{name}"] = agg
    print(f"WRN CIFAR-10-C {name} (5 seeds): clean {agg['clean']:.1f} | " + " | ".join(f"{c} {np.mean(agg[c]):.1f}" for c in NO)
          + f" | mean of gaussian/shot/impulse {agg['noise3_mean']:.1f}", flush=True)

# ---- (2) MNIST-C
mc = {c: (np.load(f"{OFF}/mnist_c/{c}/test_images.npy"), np.load(f"{OFF}/mnist_c/{c}/test_labels.npy")) for c in ("identity", "shot_noise", "impulse_noise")}
RUNS = {"FCN": ["fcn_mnist_tanh_ns_ki_d10"] + [f"fcn_mnist_tanh_ns_ki_d10_s{s}" for s in (43, 44, 45, 46)],
        "CNN": ["cnn_mnist_perch"] + [f"cnn_mnist_perch_s{s}" for s in (1, 2, 3, 4)]}
for arch, runs in RUNS.items():
    for k, name in ((0, "with LI"), (1, "without LI")):
        rows = []
        for run in runs:
            m, meta, _ = load(run, k)
            rows.append([acc_u8(m, mc[c][0], mc[c][1], meta["mean"], meta["std"]) for c in mc])
        a = np.array(rows); out[f"MNISTC_{arch}|{name}"] = {c: (float(a[:, i].mean()), float(a[:, i].std())) for i, c in enumerate(mc)}
        print(f"MNIST-C {arch} {name} (5 seeds): " + " | ".join(f"{c} {a[:, i].mean():.1f}±{a[:, i].std():.1f}" for i, c in enumerate(mc)), flush=True)

# ---- (3) MNIST-C noise strength on our ladder
idn = mc["identity"][0][:1000, :, :, 0].astype(float) / 255
for c in ("shot_noise", "impulse_noise"):
    d = np.abs(mc[c][0][:1000, :, :, 0].astype(float) / 255 - idn).mean()
    Xs = torch.from_numpy((idn - 0.5) / 0.5)[:, None].float()
    ours = [float(((_corrupt_batch(Xs, torch.tensor([0.5]), torch.tensor([0.5]), c, l) * 0.5 + 0.5)[:, 0].numpy() - idn).__abs__().mean()) for l in range(9)]
    out[f"MNISTC_strength|{c}"] = {"mnistc": float(d), "ours_by_level": ours}
    print(f"{c}: MNIST-C mean |change| per pixel {d:.4f}; ours at levels 1-9: " + " ".join(f"{v:.4f}" for v in ours))
json.dump(out, open(f"{ROOT}/diagnostics/official_c_noise_eval.json", "w"), indent=2)
print("WROTE diagnostics/official_c_noise_eval.json")
