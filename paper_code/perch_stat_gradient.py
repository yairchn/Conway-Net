"""Reviewer P3: the image gate's per-channel statistics (median, contrast, contrast weights; _perch_bgref_gates) are
detached in the backward pass, so the input gradient of the CNN is a fixed-statistic surrogate, not the derivative of
the deployed forward function. Inference only; training is unchanged. Three parts, each against the exact derivative
("exact": the same forward with the statistics left in the graph; torch's median / amax pass the gradient to the
selected element, the derivative wherever the median and maximum are unique):
 (A) the gate alone: scaling a map's contrast about its median leaves the occupancy unchanged; finite difference vs
     autograd, detached and exact, of sum(B + S) along that direction and along a random one.
 (B) trained CNNs with lateral inhibition (8 MNIST-family datasets, grayscale and colour, 5 seeds; first 1000 test
     images, true class): saliency with the exact derivative vs the surrogate (the paper's): the signed log-ratio and
     share of images favouring the object (as saliency_signed_metrics.py), per-image Spearman correlation of the two
     maps and their relative L2 difference; and, in float64 on 50 images of each seed-0 network, the directional
     derivative of logit_y along a random unit direction and along contrast scaling of the image about its
     per-channel median, from autograd (surrogate, exact) and from a central finite difference.
 (C) constant and near-constant images, where the contrast safeguard (eps = 1e-6) governs the gate: background 0.5
     plus uniform noise of amplitude a, and a test digit with its contrast about 0.5 scaled by a; per a: the gate's
     contrast, the mean birth and survival gates, the share of images assigned to the modal class, the mean max
     softmax, and the input-gradient norm (surrogate, exact).
Writes diagnostics/perch_stat_gradient.json. Run: PYTHONPATH=. python paper_code/perch_stat_gradient.py
"""
from conwaynet import REPO_ROOT
import importlib
import json

import numpy as np
import torch
from scipy import ndimage
from scipy.stats import spearmanr

LIM = importlib.import_module("conwaynet.lateral_inhibition")
_src = open(f"{REPO_ROOT}/paper_code/saliency_signed_metrics.py").read()
exec(_src[_src.index("def otsu("):_src.index("def logits(")])        # otsu, mask_of
from conwaynet.config import load_config
from conwaynet.data import load_dataset
from conwaynet.models import build_model

ROOT = REPO_ROOT; DEV = "cuda"; N = 1000; N_FD = 50
MAP = json.load(open(f"{ROOT}/paper_runs/_paper_net_configs.json"))
DATASETS = ("mnist", "fashionmnist", "kmnist", "emnist", "mnist_color", "fashionmnist_color", "kmnist_color", "emnist_color")
SURROGATE = LIM._perch_bgref_gates


def exact(m, births, survivals, delta, gamma, eps=1e-6):
    """_perch_bgref_gates with the statistics left in the graph (identical forward)."""
    flat = m.flatten(2)
    med = flat.median(dim=-1, keepdim=True).values
    dev = (flat - med).abs()
    con = dev.amax(dim=-1, keepdim=True)
    occ = (dev / con.clamp_min(eps)).view_as(m)
    w = (con / con.sum(dim=1, keepdim=True).clamp_min(eps)).unsqueeze(-1)
    return ((LIM._rule_mask(occ, births, delta, gamma) * w).sum(1, keepdim=True),
            (LIM._rule_mask(occ, survivals, delta, gamma) * w).sum(1, keepdim=True))


def use(mode):
    LIM._perch_bgref_gates = exact if mode == "exact" else SURROGATE


res = {}
# ------------------------------------------------------------------------------------------------ (A) gate alone
torch.manual_seed(0)
B5, S3 = LIM.parse_rule("B5/S3")
m0 = (torch.rand(1, 1, 28, 28, dtype=torch.float64) * 0.2 + torch.nn.functional.pad(torch.ones(1, 1, 10, 10, dtype=torch.float64) * 0.6, (9, 9, 9, 9)))
med0 = m0.flatten(2).median(-1).values[..., None, None]
dirs = {"contrast about the median": m0 - med0, "random": torch.randn_like(m0)}
res["A"] = {}
for name, d in dirs.items():
    d = d / d.norm(); row = {}
    for mode in ("surrogate", "exact"):
        use(mode); x = m0.clone().requires_grad_(True); b, s = LIM._perch_bgref_gates(x, B5, S3, 0.5, 15.0); (b + s).sum().backward()
        row[f"autograd_{mode}"] = float((x.grad * d).sum())
    use("surrogate"); h = 1e-6
    f = lambda t: sum(v.sum() for v in LIM._perch_bgref_gates(t, B5, S3, 0.5, 15.0))
    row["finite_difference"] = float((f(m0 + h * d) - f(m0 - h * d)) / (2 * h))
    res["A"][name] = row
    print(f"(A) {name:26s} FD {row['finite_difference']:+.4f}  autograd surrogate {row['autograd_surrogate']:+.4f}  exact {row['autograd_exact']:+.4f}", flush=True)


# ------------------------------------------------------------------------------------------------ (B) trained CNNs
def load(run):
    cfg = load_config(MAP[run][0])
    _, _, te, meta = load_dataset(cfg["data_path"], 500, cfg["data_name"], val_frac=cfg["val_frac"], split_seed=cfg["split_seed"],
                                  data_norms=cfg["data_norms"], output_path=f"{ROOT}/paper_runs/{run}", num_workers=2)
    X, Y = [], []
    for xb, yb in te:
        X.append(xb); Y.append(yb)
        if sum(len(x) for x in X) >= N: break
    X, Y = torch.cat(X)[:N], torch.cat(Y)[:N]
    m = build_model(cfg, meta); m.load_state_dict(torch.load(f"{ROOT}/paper_runs/{run}/model_ca.pth", map_location="cpu")); m.to(DEV).eval()
    return m, X, Y, meta


def saliency(m, X, T, mode, dtype=torch.float32):
    use(mode); out = []
    for i in range(0, len(X), 250):
        x = X[i:i + 250].clone().to(DEV, dtype).requires_grad_(True)
        m(x).gather(1, T[i:i + 250].to(DEV)[:, None]).sum().backward()
        out.append(x.grad.abs().sum(1).double().cpu().numpy())
    use("surrogate")
    return np.concatenate(out)


def signed(S, masks):
    on = np.array([g[k].mean() for g, k in zip(S, masks)]); off = np.array([g[~k].mean() for g, k in zip(S, masks)])
    return float(np.mean(np.log10(on / off))), float(np.mean(on > off))


def fd_check(m, X, Y, meta):
    """Directional derivative of logit_y, float64: autograd (surrogate, exact) vs central finite difference."""
    m = m.double(); out = {}
    x0 = X[:N_FD].to(DEV, torch.float64); y = Y[:N_FD].to(DEV)
    g = torch.Generator().manual_seed(0)
    med = x0.flatten(2).median(-1).values[..., None, None]
    dirs = {"random": torch.randn(x0.shape, generator=g, dtype=torch.float64).to(DEV), "contrast about the median": x0 - med}
    f = lambda t: m(t).gather(1, y[:, None])[:, 0]
    for name, d in dirs.items():
        d = d / d.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
        with torch.no_grad():
            fd = ((f(x0 + 1e-5 * d) - f(x0 - 1e-5 * d)) / 2e-5).cpu().numpy()
        row = {}
        for mode in ("surrogate", "exact"):
            use(mode); x = x0.clone().requires_grad_(True); f(x).sum().backward()
            ad = (x.grad * d).flatten(1).sum(1).cpu().numpy(); use("surrogate")
            rel = np.abs(ad - fd) / np.maximum(np.abs(fd), 1e-12)
            row[mode] = dict(median_rel_error=float(np.median(rel)), frac_rel_error_below_1pct=float(np.mean(rel < 0.01)),
                             median_abs_fd=float(np.median(np.abs(fd))))
        out[name] = row
    m.float()
    return out


res["B"] = {}
for ds in DATASETS:
    runs = [f"cnn_{ds}_perch"] + [f"cnn_{ds}_perch_s{s}" for s in (1, 2, 3, 4)]
    rows = []
    for k, run in enumerate(runs):
        m, X, Y, meta = load(run)
        mn = torch.tensor(meta["mean"]).view(1, -1, 1, 1); sd = torch.tensor(meta["std"]).view(1, -1, 1, 1)
        masks = [mask_of(im) for im in (X * sd + mn).clamp(0, 1).numpy()]
        Ss, Se = saliency(m, X, Y, "surrogate"), saliency(m, X, Y, "exact")
        rho = [spearmanr(a.ravel(), b.ravel())[0] for a, b in zip(Ss, Se)]
        rel = np.linalg.norm((Ss - Se).reshape(N, -1), axis=1) / np.linalg.norm(Ss.reshape(N, -1), axis=1)
        (ls, fs), (le, fe) = signed(Ss, masks), signed(Se, masks)
        row = dict(log_ratio_surrogate=ls, log_ratio_exact=le, favours_surrogate=fs, favours_exact=fe,
                   spearman_median=float(np.median(rho)), spearman_min=float(np.min(rho)), rel_l2_median=float(np.median(rel)))
        if k == 0:
            row["fd"] = fd_check(m, X, Y, meta)
        rows.append(row)
    agg = {key: (float(np.mean([r[key] for r in rows])), float(np.std([r[key] for r in rows]))) for key in rows[0] if key != "fd"}
    agg["fd_seed0"] = rows[0]["fd"]; res["B"][ds] = agg
    fd = rows[0]["fd"]
    print(f"(B) {ds:20s} log-ratio surrogate {agg['log_ratio_surrogate'][0]:+.2f} exact {agg['log_ratio_exact'][0]:+.2f} | favours {agg['favours_surrogate'][0]:.2f} / {agg['favours_exact'][0]:.2f} | "
          f"Spearman {agg['spearman_median'][0]:.3f} (min {agg['spearman_min'][0]:.2f}) | rel L2 {agg['rel_l2_median'][0]:.3f} | FD rel err (random dir) surrogate "
          f"{fd['random']['surrogate']['median_rel_error']:.2e} exact {fd['random']['exact']['median_rel_error']:.2e}; (contrast dir) surrogate "
          f"{fd['contrast about the median']['surrogate']['median_rel_error']:.2e} exact {fd['contrast about the median']['exact']['median_rel_error']:.2e}", flush=True)

# ------------------------------------------------------------------------------------------------ (C) near-constant images
res["C"] = {}
AMPS = [0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
for ds in ("mnist", "mnist_color"):
    m, X, Y, meta = load(f"cnn_{ds}_perch")
    mn = torch.tensor(meta["mean"]).view(1, -1, 1, 1); sd = torch.tensor(meta["std"]).view(1, -1, 1, 1)
    img01 = (X[:200] * sd + mn).clamp(0, 1); C = img01.shape[1]
    stats = {}
    hook = m.li1.register_forward_hook(lambda mod, inp, out: stats.__setitem__("x", inp[0].detach()))
    g = torch.Generator().manual_seed(0)
    rows = {}
    for kind in ("background 0.5 + uniform noise of amplitude a", "test digit, contrast about 0.5 scaled by a"):
        for a in AMPS:
            if kind.startswith("background"):
                u = 0.5 + a * (2 * torch.rand((200, C, 28, 28), generator=g) - 1)
            else:
                u = 0.5 + a * (img01 - 0.5)
            x = ((u - mn) / sd).float()
            row = {}
            for mode in ("surrogate", "exact"):
                use(mode); xx = x.clone().to(DEV).requires_grad_(True); out = m(xx)
                out.max(1).values.sum().backward(); use("surrogate")
                row[f"grad_norm_{mode}"] = float(xx.grad.flatten(1).norm(dim=1).median())
            with torch.no_grad():
                p = torch.softmax(out.detach(), 1); pred = p.argmax(1).cpu()
                xm = LIM.neighbour_mean((stats["x"] + 1) / 2, 1).flatten(2)
                con = (xm - xm.median(-1, keepdim=True).values).abs().amax(-1)
                b, s = SURROGATE(LIM.neighbour_mean((stats["x"] + 1) / 2, 1), *LIM.parse_rule("B5/S3"), 0.5, 15.0)
            row.update(contrast_median=float(con.max(1).values.median()), birth_mean=float(b.mean()), survive_mean=float(s.mean()),
                       modal_class_share=float(np.bincount(pred.numpy()).max() / len(pred)), max_softmax_mean=float(p.max(1).values.mean()),
                       acc=float((pred == Y[:200]).float().mean()) if kind.startswith("test") else None)
            rows[f"{kind} | a={a:g}"] = row
            print(f"(C) {ds:12s} {kind[:30]:30s} a={a:7.0e}  contrast {row['contrast_median']:.1e}  B {row['birth_mean']:.3f} S {row['survive_mean']:.3f}  "
                  f"modal class {row['modal_class_share']:.2f}  max p {row['max_softmax_mean']:.2f}  acc {row['acc']}  "
                  f"|grad| surrogate {row['grad_norm_surrogate']:.2e} exact {row['grad_norm_exact']:.2e}", flush=True)
    hook.remove(); res["C"][ds] = rows
json.dump(res, open(f"{ROOT}/diagnostics/perch_stat_gradient.json", "w"), indent=1)
print("WROTE diagnostics/perch_stat_gradient.json")
