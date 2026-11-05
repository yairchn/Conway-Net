"""Gradient-saliency on/off-object SEPARATION across selected paper-net samples, LI=CA vs none.

For each sample: gradient saliency g=|d logit_t/dx| (channel-summed). Object mask = Moore-dilated
non-background input pixels. on = mean g on-object, off = mean g off-object. SEPARATION score
  sep = max(on/off, off/on)   (>=1; =1 no separation; high = clean object/bg separation EITHER polarity)
— symmetric so a background-classifier (off>>on) scores as high as an on-object one; low only when the
map cannot separate object from background. The figure annotates the SIGNED log10(on/off) (> 0: object), which,
unlike sep, tells object from background reliance (reviewer P2). Figure: input | grad CA | grad none, one row per sample.
Run: PYTHONPATH=. python diagnostics/saliency_onoff_separation.py
"""
from conwaynet import REPO_ROOT
import json, numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import ndimage
from conwaynet.config import load_config
from conwaynet.models import build_model
from conwaynet.data import load_dataset
DEV = "cuda" if torch.cuda.is_available() else "cpu"; ROOT = REPO_ROOT
MAP = json.load(open(f"{ROOT}/paper_runs/_paper_net_configs.json"))
# Two blocks, split by modality. Left = grayscale strokes (incl. grayscale WRN-EMNIST);
# right = colour textures (colour CNNs + WRN-CIFAR-10).
GRAY = [("FCN","mnist","fcn_mnist_tanh_ns_ki_d10",7),
        ("FCN","fashion","fcn_fashionmnist_tanh_ns_ki_d10",1),
        ("FCN","kmnist","fcn_kmnist_tanh_ns_ki_d10",9),
        ("FCN","emnist","fcn_emnist_tanh_ns_ki_d10",26),
        ("CNN","mnist","cnn_mnist_perch",7),
        ("CNN","fashion","cnn_fashionmnist_perch",1)]
COLOUR = [("CNN","mnist-c","cnn_mnist_color_perch",7),
          ("CNN","fashion-c","cnn_fashionmnist_color_perch",2),
          ("CNN","kmnist-c","cnn_kmnist_color_perch",3),
          ("CNN","emnist-c","cnn_emnist_color_perch",1),
          ("WRN","cifar10","wrn_cifar10",0),
          ("WRN","emnist","wrn_emnist",1)]

def load(run, mech):
    cfg = load_config(MAP[run][0 if mech=="ca" else 1]); rd = f"{ROOT}/paper_runs/{run}"
    tr, va, te, meta = load_dataset(cfg["data_path"], cfg["batch_size"], cfg["data_name"], val_frac=cfg["val_frac"],
        split_seed=cfg["split_seed"], data_norms=cfg["data_norms"], output_path=rd, num_workers=2)
    m = build_model(cfg, meta).to(DEV); m.load_state_dict(torch.load(f"{rd}/model_{mech}.pth", map_location=DEV)); m.eval()
    return m, meta, te

def out_(m, x):
    o = m(x); return o[0] if isinstance(o, tuple) else o

def grad_map(m, x, t):
    x = x.clone().requires_grad_(True); out_(m, x)[0, t].backward()
    return x.grad.detach()[0].abs().sum(0).cpu().numpy()

def find_sample(te, cls, meta, n=4000):
    xs, ys = [], []
    for xb, yb in te:
        xs.append(xb); ys.append(yb)
        if sum(len(z) for z in xs) >= n: break
    X = torch.cat(xs); Y = torch.cat(ys); idx = (Y == cls).nonzero(as_tuple=True)[0]
    if len(idx) == 0: idx = (Y == int(Y.max())).nonzero(as_tuple=True)[0]  # fallback
    return X[idx[0]:idx[0]+1]

def _otsu(x):
    h, e = np.histogram(x.ravel(), 256, (0, 1)); p = h.astype(float)/max(h.sum(),1)
    om = np.cumsum(p); mu = np.cumsum(p*((e[:-1]+e[1:])/2)); muT = mu[-1]
    sig = (muT*om - mu)**2 / (om*(1-om) + 1e-12)
    return float((e[:-1]+e[1:])[np.nanargmax(sig)]/2)

def obj_mask(img01):
    """Moore-dilated object mask. Black-background grayscale: luminance>0.1. Colour/no-black-bg: Otsu."""
    if img01.ndim == 3 and img01.shape[0] > 1:        # colour -> Otsu split (no clean black bg)
        lum = img01.mean(0); thr = _otsu(lum)
    else:
        lum = img01[0] if img01.ndim == 3 else img01; thr = 0.1
    return ndimage.binary_dilation(lum > thr, iterations=1)

def sep_score(g, mask):
    onf = mask.mean()
    if onf < 0.03 or onf > 0.97:                       # degenerate: frame-filling / no background
        return float("nan"), float(g[mask].mean()) if mask.any() else 0.0, float(g[~mask].mean()) if (~mask).any() else 0.0
    on = max(float(g[mask].mean()), 1e-9); off = max(float(g[~mask].mean()), 1e-9)
    return max(on/off, off/on), on, off

NC = 3                              # columns per block (input | grad CA | grad none)
nrows = max(len(GRAY), len(COLOUR))
fig, axes = plt.subplots(nrows, 2*NC, figsize=(15.5, 1.95*nrows), squeeze=False)

def draw_row(r, c0, arch, ds, run, cls):
    m_ca, meta, te = load(run, "ca"); m_no, _, _ = load(run, "none")
    mn = torch.tensor(meta["mean"]).view(-1,1,1); sd = torch.tensor(meta["std"]).view(-1,1,1)
    x = find_sample(te, cls, meta).to(DEV)
    img01 = (x[0].cpu()*sd+mn).clamp(0,1).numpy()
    mask = obj_mask(img01)
    gca, gno = grad_map(m_ca, x, cls), grad_map(m_no, x, cls)
    sca, onca, offca = sep_score(gca, mask); sno, onno, offno = sep_score(gno, mask)
    real_colour = (run == "wrn_cifar10")   # real photo: no clean object/background mask, no sep number
    disp = img01[0] if img01.shape[0]==1 else np.transpose(img01,(1,2,0))
    axes[r][c0].imshow(disp, cmap="gray" if img01.shape[0]==1 else None)
    if not real_colour:
        axes[r][c0].contour(mask, levels=[0.5], colors="lime", linewidths=0.6)
    axes[r][c0+1].imshow(gca, cmap="magma", vmin=0, vmax=np.percentile(gca,99.5) or 1)
    axes[r][c0+2].imshow(gno, cmap="magma", vmin=0, vmax=np.percentile(gno,99.5) or 1)
    for c in range(NC): axes[r][c0+c].axis("off")
    axes[r][c0].text(-0.10,0.5,f"{arch.replace('FCN', 'MLP')}\n{ds}\ncls {cls}", transform=axes[r][c0].transAxes, rotation=0, va="center", ha="right", fontsize=7.5)
    if real_colour:
        axes[r][c0+1].set_title("with lateral inhibition", fontsize=8, color="#555"); axes[r][c0+2].set_title("plain", fontsize=8, color="#555")
    else:
        lca, lno = np.log10(onca / offca), np.log10(onno / offno)     # signed: > 0 object, < 0 background
        axes[r][c0+1].set_title(f"with lateral inhibition\nlog$_{{10}}$ on/off = {lca:+.2f}", fontsize=8.5, color="#333")
        axes[r][c0+2].set_title(f"plain\nlog$_{{10}}$ on/off = {lno:+.2f}", fontsize=8.5, color="#333")
    if r==0: axes[0][c0].set_title("input", fontsize=8.5)
    print(f"{arch}/{ds} cls{cls}: CA sep={sca:.2f} (on {onca:.3f}/off {offca:.3f}) | none sep={sno:.2f} (on {onno:.3f}/off {offno:.3f})", flush=True)
    del m_ca, m_no; torch.cuda.empty_cache()

for r in range(nrows):
    if r < len(GRAY): draw_row(r, 0, *GRAY[r])
    else:
        for c in range(NC): axes[r][c].axis("off")
    if r < len(COLOUR): draw_row(r, NC, *COLOUR[r])
    else:
        for c in range(NC): axes[r][NC+c].axis("off")

fig.text(0.26, 0.995, "grayscale  —  strokes", ha="center", va="top", fontsize=14, weight="bold")
fig.text(0.76, 0.995, "colour  —  textures", ha="center", va="top", fontsize=14, weight="bold")
fig.subplots_adjust(wspace=0.08)
out=f"{ROOT}/paper_figures/out/saliency_onoff_separation.png"
fig.tight_layout(rect=[0,0,1,0.965]); fig.savefig(out, dpi=140, bbox_inches="tight"); print("WROTE", out)
