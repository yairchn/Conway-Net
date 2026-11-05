"""FCN-10 depth diagnostics (clean paper nets) — two complementary views of how the representation
evolves through depth, combined in one figure.  LI=CA vs LI=none, MNIST, fcn_mnist_tanh_ns_ki_d10.

  (1) LOGIT-LENS — apply the TRAINED final classifier to every intermediate layer's activation and
      score test accuracy.  Asks: how early in depth is the output already linearly decodable in the
      classifier's own frame?  (A class-aligned representation is readable long before the last layer.)
  (2) SCALE SEPARATION — treat depth as time; band-pass each layer's activation map by spatial
      frequency and measure the across-layer correlation time tau per band.  Asks: is there a slow
      large-scale carrier with fast small-scale churn underneath?  SSI = tau_coarse / tau_fine.

The two overlap: the slowly-evolving coarse scale is the class-predictive content the logit-lens reads
and holds through depth, while the fast fine scale is detail being refined and is invisible to the
linear readout.

Run:  PYTHONPATH=. python diagnostics/fcn_depth_diagnostics.py
Out:  paper_figures/out/fcn_depth_diagnostics.png  (+ printed numbers)
"""
from conwaynet import REPO_ROOT
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import sys
sys.path.insert(0, f"{REPO_ROOT}/paper_code")
import scale_separation_index as SS
from conwaynet.config import load_config
from conwaynet.models import build_model
from conwaynet.data import load_dataset
DEV = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = REPO_ROOT; HW = 28; N = 2000
CFG = f"{ROOT}/configs/paper/tanh/fcn_mnist_tanh_ns_ki_{{}}.json"; RUN = f"{ROOT}/paper_runs/fcn_mnist_tanh_ns_ki_d10"
CA, NO = "#d1495b", "#30638e"
BANDLAB = ["0-2\n(coarse)", "2-4", "4-6", "6-9", "9-13", "13+\n(fine)"]
BCOL = plt.cm.viridis(np.linspace(0, 0.9, len(BANDLAB)))

def load(mech):
    c = load_config(CFG.format(mech))
    tr, va, te, meta = load_dataset(c["data_path"], c["batch_size"], c["data_name"], val_frac=c["val_frac"],
        split_seed=c["split_seed"], data_norms=c["data_norms"], output_path=RUN, num_workers=2)
    m = build_model(c, meta).to(DEV); m.load_state_dict(torch.load(f"{RUN}/model_{mech}.pth", map_location=DEV)); m.eval()
    xs, ys = [], []
    for xb, yb in te:
        xs.append(xb); ys.append(yb)
        if sum(len(z) for z in xs) >= N: break
    return m, torch.cat(xs)[:N].to(DEV), torch.cat(ys)[:N].to(DEV)

@torch.no_grad()
def forward_layers(m, X):
    """Return per-layer activations cur_ℓ (N, 784) after each (linear+norm+act+gate) block."""
    cur = X.view(X.size(0), -1); outs = []
    for i, lin in enumerate(m.linears):
        t = lin(cur); t = m.norms[i](t) if m.norms is not None else t; z = torch.tanh(t)
        if m.gate is not None: z = m.gate(cur, z)
        cur = z; outs.append(cur)
    return outs

@torch.no_grad()
def logit_lens(m, X, Y):
    """Accuracy of the TRAINED final classifier applied to each intermediate layer."""
    accs = []
    for cur in forward_layers(m, X):
        pred = m.classifier(cur).argmax(1)
        accs.append(float((pred == Y).float().mean()) * 100)
    return accs

def scale_sep(m, X):
    """Per-band tau and C_b(k), the shared definition (paper_code/scale_separation_index.py)."""
    r = SS.scale_separation(torch.stack([c.view(c.size(0), HW, HW) for c in forward_layers(m, X[:SS.N_IMAGES])]))
    return r["tau"], r["C"], r["censored"]

RES = {}
for mech in ("ca", "none"):
    m, X, Y = load(mech)
    ll = logit_lens(m, X, Y); taus, Cs, cens = scale_sep(m, X)
    ssi = taus[0]/taus[-1]
    RES[mech] = dict(ll=ll, taus=taus, Cs=Cs, cens=cens, ssi=ssi)
    print(f"{mech}: logit-lens acc/layer={[round(a) for a in ll]} | SSI={ssi:.2f} tau={[round(t,1) for t in taus]}", flush=True)
    del m; torch.cuda.empty_cache()

# ---------------- combined figure ----------------
fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
# Panel A: logit-lens
a = axes[0]; L = len(RES["ca"]["ll"]); xl = np.arange(1, L+1)
a.plot(xl, RES["ca"]["ll"], "-o", color=CA, lw=2, label="with lateral inhibition")
a.plot(xl, RES["none"]["ll"], "-o", color=NO, lw=2, label="without lateral inhibition")
a.set_xlabel("layer (depth)", fontsize=10); a.set_ylabel("final-classifier accuracy (%)", fontsize=10)

a.legend(fontsize=9, frameon=False); a.grid(alpha=0.25); a.set_ylim(0, 100)
# Panel B: scale-sep tau vs band
a = axes[1]; bx = np.arange(len(BANDLAB))
for mech, col, lab in (("ca", CA, "with lateral inhibition"), ("none", NO, "without lateral inhibition")):
    t = np.array(RES[mech]["taus"]); cz = np.array(RES[mech]["cens"])
    a.plot(bx, t, "-", color=col, lw=2, label=f"{lab} (SSI {RES[mech]['ssi']:.1f})")
    a.plot(bx[~cz], t[~cz], "o", color=col, ms=6); a.plot(bx[cz], t[cz], "^", color=col, mfc="white", ms=8, mew=1.6)
a.axhline(SS.MAX_LAG + 0.5, color="#555", ls="--", lw=1)
a.text(len(BANDLAB) - 1, SS.MAX_LAG + 0.5, "horizon: τ ≤ 9.5", ha="right", va="bottom", fontsize=8, color="#555")
a.plot([], [], "^", color="#555", mfc="white", ms=8, mew=1.6, ls="none", label="censored (C > 0 at every lag)")
a.set_xticks(bx); a.set_xticklabels(BANDLAB, fontsize=7.5)
a.set_xlabel("spatial-frequency band", fontsize=10); a.set_ylabel("correlation depth τ (layers)", fontsize=10)

a.legend(fontsize=8, frameon=False, loc="center right"); a.grid(alpha=0.25); a.set_ylim(0, 10.5)
# Panel C: C_b(k) for CA (the overlap: coarse persists = the decodable carrier)
a = axes[2]
for bi in range(len(BANDLAB)):
    C = RES["ca"]["Cs"][bi]; a.plot(range(len(C)), C, "-o", ms=3, color=BCOL[bi], label=BANDLAB[bi].replace("\n"," "))
a.set_xlabel("depth lag k (layers)", fontsize=10); a.set_ylabel("across-layer corr C_b(k)", fontsize=10)

a.legend(fontsize=7, frameon=False, ncol=2); a.grid(alpha=0.25); a.set_ylim(-0.3, 1.02); a.axhline(0, color="#999", lw=0.8)
for ax in axes:
    for s in ("top","right"): ax.spines[s].set_visible(False)

fig.tight_layout(rect=[0,0,1,1]); out = f"{ROOT}/paper_figures/out/fcn_depth_diagnostics.png"; fig.savefig(out, dpi=140)
print("WROTE", out)
