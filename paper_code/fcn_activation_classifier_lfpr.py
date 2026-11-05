"""Paper FCN (MNIST, clean ki) — class-averaged ACTIVATION maps (last hidden layer) + CLASSIFIER
weights, LI=CA vs none; and LFPR (low-frequency power ratio) of the activation maps, mean±std across
the 10 digit classes. LFPR = central low-freq power / total power (DC removed, radius 0.25): higher =
smoother/more coherent (digit-lit) vs salt-and-pepper. Outputs two figures in diagnostics/."""
from conwaynet import REPO_ROOT
import numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats as sstats
from conwaynet.config import load_config
from conwaynet.models import build_model
from conwaynet.data import load_dataset
DEV = "cuda" if torch.cuda.is_available() else "cpu"; ROOT = REPO_ROOT; N = 4000; HW = 28
CA, NO = "#d1495b", "#30638e"; RUN = f"{ROOT}/paper_runs/fcn_mnist_tanh_ns_ki_d10"

def load(mech):
    c = load_config(f"{ROOT}/configs/paper/tanh/fcn_mnist_tanh_ns_ki_{mech}.json")
    tr, va, te, meta = load_dataset(c["data_path"], c["batch_size"], c["data_name"], val_frac=c["val_frac"],
        split_seed=c["split_seed"], data_norms=c["data_norms"], output_path=RUN, num_workers=2)
    m = build_model(c, meta).to(DEV); m.load_state_dict(torch.load(f"{RUN}/model_{mech}.pth", map_location=DEV)); m.eval()
    xs, ys = [], []
    for xb, yb in te:
        xs.append(xb); ys.append(yb)
        if sum(len(z) for z in xs) >= N: break
    return m, torch.cat(xs)[:N].to(DEV), torch.cat(ys)[:N].numpy()

@torch.no_grad()
def last_hidden(m, X, bs=256):
    out = []
    for i in range(0, len(X), bs):
        cur = X[i:i+bs].view(X[i:i+bs].size(0), -1)
        for j, lin in enumerate(m.linears):
            t = lin(cur); t = m.norms[j](t) if m.norms is not None else t; z = torch.tanh(t)
            if m.gate is not None: z = m.gate(cur, z)
            cur = z
        out.append(cur.cpu())
    return torch.cat(out).numpy()

def class_avg_maps(F, Y):
    return np.stack([F[Y == c].mean(0).reshape(HW, HW) for c in range(10)])   # (10,28,28)

def lfpr(W, radius_frac=0.25, eps=1e-8):
    x = W - W.mean(axis=(-2,-1), keepdims=True)
    P = np.abs(np.fft.fftshift(np.fft.fft2(x, axes=(-2,-1)), axes=(-2,-1)))**2
    yy, xx = np.meshgrid(np.arange(HW), np.arange(HW), indexing="ij")
    r = np.sqrt((yy-(HW-1)/2)**2 + (xx-(HW-1)/2)**2); r = r/r.max()
    mask = (r <= radius_frac).astype(float)
    return (P*mask).sum(axis=(-2,-1)) / (P.sum(axis=(-2,-1)) + eps)

D = {}
for mech in ("ca", "none"):
    m, X, Y = load(mech); F = last_hidden(m, X)
    amaps = class_avg_maps(F, Y)
    cw = m.classifier.weight.detach().cpu().numpy().reshape(-1, HW, HW)
    D[mech] = dict(act=amaps, cw=cw, lfpr_act=lfpr(amaps), lfpr_cw=lfpr(cw))
    print(f"{mech}: LFPR activation mean={D[mech]['lfpr_act'].mean():.3f}±{D[mech]['lfpr_act'].std():.3f} | "
          f"LFPR classifier mean={D[mech]['lfpr_cw'].mean():.3f}±{D[mech]['lfpr_cw'].std():.3f}", flush=True)
    del m; torch.cuda.empty_cache()

# ---- Figure 1: activation maps + classifier weights, CA & none (4 rows x 10) ----
fig, axes = plt.subplots(4, 10, figsize=(14, 6))
rows = [("ca","act","with lateral inhibition: activation"),("none","act","without lateral inhibition: activation"),
        ("ca","cw","with lateral inhibition: classifier wt"),("none","cw","without lateral inhibition: classifier wt")]
for r,(mech,key,lab) in enumerate(rows):
    for c in range(10):
        a = D[mech][key][c]; vm = np.abs(a).max() or 1
        axes[r][c].imshow(a, cmap="RdBu_r", vmin=-vm, vmax=vm); axes[r][c].axis("off")
        if r==0: axes[0][c].set_title(str(c), fontsize=10)
    axes[r][0].text(-0.45,0.5,lab, transform=axes[r][0].transAxes, rotation=90, va="center", ha="center", fontsize=9, weight="bold")

fig.tight_layout(rect=[0,0,1,1]); fig.savefig(f"{ROOT}/diagnostics/fcn_activation_classifier_maps.png", dpi=140)
print("WROTE diagnostics/fcn_activation_classifier_maps.png")

# ---- Figure 2: LFPR — two panels (activation maps | classifier weights), mean±std across digits ----
def panel(ax, lca, lno):
    ax.bar([0,1], [lca.mean(), lno.mean()], yerr=[lca.std(), lno.std()], color=[CA, NO], capsize=6, width=0.6, error_kw=dict(lw=1.5))
    ax.set_xticks([0,1]); ax.set_xticklabels(["with lateral\ninhibition","without lateral\ninhibition"], fontsize=11); ax.set_ylim(0, 1.05)
    for i, v, s in [(0, lca.mean(), lca.std()), (1, lno.mean(), lno.std())]:
        ax.text(i, v+s+0.015, f"{v:.3f}", ha="center", fontsize=10, weight="bold")
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
panel(a1, D["ca"]["lfpr_act"], D["none"]["lfpr_act"])
panel(a2, D["ca"]["lfpr_cw"], D["none"]["lfpr_cw"])
a1.set_ylabel("LFPR (low-freq power ratio; ↑ = smoother/coherent)", fontsize=10)
fig.tight_layout(); fig.savefig(f"{ROOT}/diagnostics/fcn_lfpr_twopanel.png", dpi=140)
print("WROTE diagnostics/fcn_lfpr_twopanel.png")
