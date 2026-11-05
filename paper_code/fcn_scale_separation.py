"""Depth multi-scale (scale-separation) diagnostic for the FCN, treating DEPTH as time.

For each spatial-frequency band b, band-pass each layer's activation map, then measure the
ACROSS-LAYER autocorrelation C_b(k) = <cos(M_b,ℓ , M_b,ℓ+k)> over depth-lag k. The correlation time
τ_b (how many layers a band's spatial pattern persists) vs band frequency tells us whether a slow
large-scale structure rides over fast small-scale churn (scale separation = τ_coarse ≫ τ_fine).
Scale-separation index SSI = τ_lowfreq / τ_highfreq.

Runs d10 (CA vs none) and d100 (CA only — d100 none was not trained). MNIST. C_b(k), tau_b and SSI: the shared
definition of paper_code/scale_separation_index.py (common horizon of 9 lags at both depths, censoring flagged)."""
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
ROOT = REPO_ROOT; N = SS.N_IMAGES
HW = 28
# radial frequency bands (fftshifted radius thresholds) and labels
BANDLAB = ["0-2 (coarse)", "2-4", "4-6", "6-9", "9-13", "13+ (fine)"]
BCOL = plt.cm.viridis(np.linspace(0, 0.9, len(BANDLAB)))

def load(cfg_path, run, mech):
    c = load_config(cfg_path)
    tr, va, te, meta = load_dataset(c["data_path"], c["batch_size"], c["data_name"], val_frac=c["val_frac"],
        split_seed=c["split_seed"], data_norms=c["data_norms"], output_path=run, num_workers=2)
    m = build_model(c, meta).to(DEV); m.load_state_dict(torch.load(f"{run}/model_{mech}.pth", map_location=DEV)); m.eval()
    X = []
    for xb, _ in te:
        X.append(xb)
        if sum(len(x) for x in X) >= N: break
    return m, torch.cat(X)[:N].to(DEV)

@torch.no_grad()
def layer_maps(m, X):
    """(L, N, 28, 28) signed post-gate activation maps."""
    cur = X.view(X.size(0), -1); maps = []
    for i, lin in enumerate(m.linears):
        t = lin(cur); t = m.norms[i](t) if m.norms is not None else t; z = torch.tanh(t)
        if m.gate is not None: z = m.gate(cur, z)
        cur = z; maps.append(z.view(z.size(0), HW, HW))
    return torch.stack(maps)   # (L,N,H,W)

@torch.no_grad()
def band_corr(maps):
    """Per band: C_b(k) for k = 0..9 (the shared definition, paper_code/scale_separation_index.py) and the per-layer
    velocity mean ||dM|| / ||M|| of the band-passed maps (same masks)."""
    r = SS.scale_separation(maps); L = maps.shape[0]
    F = torch.fft.fftshift(torch.fft.fft2(maps.double()), dim=(-2, -1)); out = {}
    for bi, mk in enumerate(SS.band_masks(maps.device)):
        v = torch.fft.ifft2(torch.fft.ifftshift(F * mk, dim=(-2, -1))).real.reshape(L, maps.shape[1], -1)
        v = v - v.mean(-1, keepdim=True)
        out[bi] = dict(C=r["C"][bi], tau=r["tau"][bi], censored=r["censored"][bi],
                       vel=float(((v[1:] - v[:-1]).norm(dim=-1) / (v[:-1].norm(dim=-1) + 1e-12)).mean()))
    return out, r["ssi"]

CFG = f"{ROOT}/configs/paper/tanh"
runs = {
    "d10": dict(run=f"{ROOT}/paper_runs/fcn_mnist_tanh_ns_ki_d10", cfg=lambda a: f"{CFG}/fcn_mnist_tanh_ns_ki_{a}.json", arms=["ca","none"]),
    "d100": dict(run=f"{ROOT}/paper_runs/fcn_mnist_tanh_ns_ki_d100", cfg=lambda a: f"{CFG}/fcn_mnist_tanh_ns_ki_d100_{a}.json", arms=["ca"]),
}
RES = {}
for tag, spec in runs.items():
    for arm in spec["arms"]:
        m, X = load(spec["cfg"](arm), spec["run"], arm)
        bc, ssi = band_corr(layer_maps(m, X))
        taus = {bi: bc[bi]["tau"] for bi in bc}
        RES[(tag, arm)] = dict(bc=bc, taus=taus, ssi=ssi)
        print(f"{tag}/{arm}: SSI(τ_coarse/τ_fine)={ssi:.2f}  τ={[round(taus[b],1) for b in range(len(BANDLAB))]}  "
              f"vel={[round(bc[b]['vel'],2) for b in range(len(BANDLAB))]}", flush=True)
        del m; torch.cuda.empty_cache()

# ---------------- figure ----------------
fig = plt.figure(figsize=(13, 8.5)); gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
def plot_C(ax, key, title):
    bc = RES[key]["bc"]
    for bi in range(len(BANDLAB)):
        C = bc[bi]["C"]; ax.plot(range(len(C)), C, "-o", ms=3, color=BCOL[bi], label=BANDLAB[bi])
    ax.axhline(1/np.e, color="#999", ls=":", lw=1); ax.axhline(0, color="#ccc", lw=0.6)
    ax.set_title(title, fontsize=10); ax.set_xlabel("depth lag k (layers)", fontsize=9); ax.set_ylabel("across-layer corr C_b(k)", fontsize=9)
    ax.set_ylim(-0.1, 1.02); ax.grid(alpha=0.25)
    for s in ("top","right"): ax.spines[s].set_visible(False)
plot_C(fig.add_subplot(gs[0,0]), ("d10","ca"), "d10 LI=CA — corr vs depth-lag");
a=fig.add_subplot(gs[0,1]); plot_C(a, ("d10","none"), "d10 LI=none"); a.legend(fontsize=6.5, frameon=False, ncol=1, loc="upper right")
# tau vs band, d10 ca vs none
a=fig.add_subplot(gs[0,2]); bx=np.arange(len(BANDLAB))
a.plot(bx, [RES[("d10","ca")]["taus"][b] for b in bx], "-o", color="#d1495b", label="CA")
a.plot(bx, [RES[("d10","none")]["taus"][b] for b in bx], "-o", color="#30638e", label="none")
a.set_xticks(bx); a.set_xticklabels([l.split()[0] for l in BANDLAB], fontsize=7, rotation=20)
a.set_xlabel("spatial-frequency band", fontsize=9); a.set_ylabel("correlation time τ (layers)", fontsize=9)
a.set_title(f"d10 — persistence vs scale\nSSI CA {RES[('d10','ca')]['ssi']:.2f} vs none {RES[('d10','none')]['ssi']:.2f}", fontsize=10)
a.legend(fontsize=8, frameon=False)
for s in ("top","right"): a.spines[s].set_visible(False); a.grid(alpha=0.25)
# d100 ca: corr (long), tau vs band, velocity vs band
plot_C(fig.add_subplot(gs[1,0]), ("d100","ca"), "d100 LI=CA — corr vs depth-lag (common horizon, 9 lags)")
a=fig.add_subplot(gs[1,1]);
a.plot(bx, [RES[("d100","ca")]["taus"][b] for b in bx], "-o", color="#d1495b")
a.set_xticks(bx); a.set_xticklabels([l.split()[0] for l in BANDLAB], fontsize=7, rotation=20)
a.set_xlabel("spatial-frequency band", fontsize=9); a.set_ylabel("correlation time τ (layers)", fontsize=9)
a.set_title(f"d100 CA — persistence vs scale\nSSI={RES[('d100','ca')]['ssi']:.2f} (coarse persists {RES[('d100','ca')]['ssi']:.1f}× longer)", fontsize=10)
for s in ("top","right"): a.spines[s].set_visible(False); a.grid(alpha=0.25)
a=fig.add_subplot(gs[1,2])
a.plot(bx, [RES[("d100","ca")]["bc"][b]["vel"] for b in bx], "-o", color="#d1495b", label="d100 CA")
a.plot(bx, [RES[("d10","ca")]["bc"][b]["vel"] for b in bx], "-o", color="#e3862a", label="d10 CA")
a.set_xticks(bx); a.set_xticklabels([l.split()[0] for l in BANDLAB], fontsize=7, rotation=20)
a.set_xlabel("spatial-frequency band", fontsize=9); a.set_ylabel("layer-to-layer velocity", fontsize=9)
a.set_title("Per-scale change rate (fine churns, coarse drifts)", fontsize=10); a.legend(fontsize=8, frameon=False)
for s in ("top","right"): a.spines[s].set_visible(False); a.grid(alpha=0.25)
fig.suptitle("FCN depth scale-separation — does a slow large-scale structure ride over fast small-scale churn?  (MNIST)", fontsize=12.5, weight="bold")
fig.tight_layout(rect=[0,0,1,0.96]); out=f"{ROOT}/diagnostics/fcn_scale_separation.png"; fig.savefig(out, dpi=135); print("WROTE", out)
