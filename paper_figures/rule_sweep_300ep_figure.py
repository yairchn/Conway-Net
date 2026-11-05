"""Figure 4: MNIST rule sweep at the paper's full FCN recipe (300 epochs), 5 seeds (42-46) per rule.

Nine outer-totalistic rules (B5/S5 excluded: it does not train) and the network without lateral inhibition,
all with the paper FCN config (configs/paper/tanh/fcn_mnist_tanh_ns_ki_*), only the rule changed:
B5/S3 = the paper FCN (fcn_mnist_tanh_ns_ki_d10{,_s43..46}); other rules = fcn_mnist_tanh_ns_ki_d10_<rule>_s42..46;
without LI = the paper FCN's no-LI networks, same seeds.
(A) classifier-weight LFPR, (B) LFPR of the class-averaged last-hidden activation maps (4000 test images),
(C) training accuracy (first 10,000 training images) and validation (out-of-sample) accuracy (the run's
own validation split), mean +- s.d. over the 5 seeds.
Run: PYTHONPATH=. python paper_figures/rule_sweep_300ep_figure.py
"""
from conwaynet import REPO_ROOT
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import cm
from conwaynet.config import load_config
from conwaynet.models import build_model
from conwaynet.data import load_dataset

ROOT = REPO_ROOT; DEV = "cuda" if torch.cuda.is_available() else "cpu"
HW = 28; N = 4000; SWEEP = f"{ROOT}/paper_runs/rule_sweep"
CA = "#d1495b"; NONE_C = "#30638e"

RULES = [  # (config tag, pretty rule string)
    ("B5S3", "B5/S3  (this work)"), ("B5S23", "B5/S23"), ("B3S3", "B3/S3"),
    ("B3S23", "B3/S23  (Game of Life)"), ("B36S23", "B36/S23  (HighLife)"),
    ("B3678S34678", "B3678/S34678  (Day & Night)"), ("B2S", "B2/S  (Seeds)"),
    ("B1357S1357", "B1357/S1357  (Replicator)"), ("B34S34", "B34/S34  (3-4 Life)"),
    ("B45678S2345", "B45678/S2345  (Walled Cities)"),
]


def lfpr(W, radius_frac=0.25, eps=1e-8):
    x = W - W.mean(axis=(-2, -1), keepdims=True)
    P = np.abs(np.fft.fftshift(np.fft.fft2(x, axes=(-2, -1)), axes=(-2, -1))) ** 2
    yy, xx = np.meshgrid(np.arange(HW), np.arange(HW), indexing="ij")
    r = np.sqrt((yy - (HW - 1) / 2) ** 2 + (xx - (HW - 1) / 2) ** 2); r = r / r.max()
    mask = (r <= radius_frac).astype(float)
    return (P * mask).sum(axis=(-2, -1)) / (P.sum(axis=(-2, -1)) + eps)


@torch.no_grad()
def last_hidden(m, X, bs=256):
    out = []
    for i in range(0, len(X), bs):
        cur = X[i:i + bs].view(X[i:i + bs].size(0), -1)
        for j, lin in enumerate(m.linears):
            t = lin(cur); t = m.norms[j](t) if m.norms is not None else t; z = torch.tanh(t)
            if m.gate is not None: z = m.gate(cur, z)
            cur = z
        out.append(cur.cpu())
    return torch.cat(out).numpy()


def class_avg_maps(F, Y):
    return np.stack([F[Y == c].mean(0).reshape(HW, HW) for c in range(10)])


# test batch (same MNIST for every rule) — load once via the B5S3 config
c0 = load_config(f"{ROOT}/configs/paper/rule_sweep/fcn_mnist_rule_B5S3_ca.json")
_, _, te, meta = load_dataset(c0["data_path"], c0["batch_size"], c0["data_name"], val_frac=c0["val_frac"],
    split_seed=c0["split_seed"], data_norms=c0["data_norms"], output_path=None, num_workers=2)
_SPLITS = {}


def run_splits(c):
    """Panel C: training accuracy on the first 10,000 training images and validation (out of sample) accuracy
    on the full val carve, both from THE RUN'S OWN split (rule_sweep_seeds runs use split_seed = seed)."""
    key = c["split_seed"]
    if key not in _SPLITS:
        tr_l, va_l, _, _ = load_dataset(c["data_path"], 1000, c["data_name"], val_frac=c["val_frac"], split_seed=key,
                                        data_norms=c["data_norms"], output_path=None, num_workers=2, shuffle_train=False)
        XT, YT = [], []
        for xb, yb in tr_l:
            XT.append(xb); YT.append(yb)
            if sum(len(z) for z in XT) >= 10000: break
        XV, YV = map(torch.cat, zip(*[(xb, yb) for xb, yb in va_l]))
        _SPLITS[key] = (torch.cat(XT)[:10000], torch.cat(YT)[:10000], XV, YV)
    return _SPLITS[key]
xs, ys = [], []
for xb, yb in te:
    xs.append(xb); ys.append(yb)
    if sum(len(z) for z in xs) >= N: break
X = torch.cat(xs)[:N].to(DEV); Y = torch.cat(ys)[:N].numpy()


SEEDS = (42, 43, 44, 45, 46)
import json, os
MAP = json.load(open(f"{ROOT}/paper_runs/_paper_net_configs.json"))


def run_of(tag, sd):
    if tag in ("B5S3", "NONE"):
        return "fcn_mnist_tanh_ns_ki_d10" if sd == 42 else f"fcn_mnist_tanh_ns_ki_d10_s{sd}"
    return f"fcn_mnist_tanh_ns_ki_d10_{tag}_s{sd}"


@torch.no_grad()
def split_acc(m, XS, YS, bs=1000):
    return float(np.mean([(m(XS[i:i + bs].to(DEV)).argmax(1).cpu() == YS[i:i + bs]).float().mean().item() * len(YS[i:i + bs])
                          for i in range(0, len(XS), bs)]) * len(range(0, len(XS), bs)) / len(XS))


def measure_ckpt(cfg, rundir, mech):
    c = load_config(cfg); m = build_model(c, meta).to(DEV)
    m.load_state_dict(torch.load(f"{rundir}/model_{mech}.pth", map_location=DEV)); m.eval()
    cw = m.classifier.weight.detach().cpu().numpy().reshape(-1, HW, HW)
    amaps = class_avg_maps(last_hidden(m, X), Y)
    lc, la = float(lfpr(cw).mean()), float(lfpr(amaps).mean())
    XT, YT, XV, YV = run_splits(c)
    tra, vaa = split_acc(m, XT, YT), split_acc(m, XV, YV)
    del m; torch.cuda.empty_cache()
    return lc, la, tra, vaa


def acc_of(rundir, mech):
    r = json.load(open(f"{rundir}/results_{mech}.json"))
    return r["test_clean_acc"]


rows = []  # (pretty, cw_mean, cw_std, act_mean, act_std, acc_mean, acc_std, train_m, train_s, val_m, val_s)
for tag, pretty in RULES:
    cws, acts, accs, trs, vas = [], [], [], [], []
    for sd in SEEDS:
        run = run_of(tag, sd); rundir = f"{ROOT}/paper_runs/{run}"; cfg = f"{ROOT}/{MAP[run][0]}"
        if not os.path.exists(f"{rundir}/model_ca.pth"):
            continue
        lc, la, ta, va = measure_ckpt(cfg, rundir, "ca"); cws.append(lc); acts.append(la); accs.append(acc_of(rundir, "ca"))
        trs.append(ta); vas.append(va)
    cws, acts, accs, trs, vas = map(np.array, (cws, acts, accs, trs, vas))
    rows.append((pretty, cws.mean(), cws.std(), acts.mean(), acts.std(), accs.mean(), accs.std(),
                 trs.mean(), trs.std(), vas.mean(), vas.std()))
    print(f"{pretty:34s} cw={cws.mean():.3f}±{cws.std():.3f}  act={acts.mean():.3f}±{acts.std():.3f}  "
          f"acc={accs.mean()*100:.1f}±{accs.std()*100:.1f}  train={trs.mean()*100:.1f} val={vas.mean()*100:.1f}  (n={len(cws)})", flush=True)
# none reference across seeds
ncws, nacts, naccs, ntrs, nvas = [], [], [], [], []
for sd in SEEDS:
    run = run_of("NONE", sd); rundir = f"{ROOT}/paper_runs/{run}"; cfg = f"{ROOT}/{MAP[run][1]}"
    if not os.path.exists(f"{rundir}/model_none.pth"):
        continue
    lc, la, ta, va = measure_ckpt(cfg, rundir, "none"); ncws.append(lc); nacts.append(la); naccs.append(acc_of(rundir, "none"))
    ntrs.append(ta); nvas.append(va)
none_cw, none_act = float(np.mean(ncws)), float(np.mean(nacts))
NONE_LABEL = "plain"
rows.append((NONE_LABEL, none_cw, float(np.std(ncws)), none_act, float(np.std(nacts)), float(np.mean(naccs)), float(np.std(naccs)),
             float(np.mean(ntrs)), float(np.std(ntrs)), float(np.mean(nvas)), float(np.std(nvas))))
print(f"{'li=none (no rule)':34s} cw={none_cw:.3f}  act={none_act:.3f}  acc={np.mean(naccs)*100:.1f}", flush=True)

rows.sort(key=lambda r: r[3])  # ascending activation-map LFPR (panel A) -> increasing up the y axis
labels = [r[0] for r in rows]
cw_m = np.array([r[1] for r in rows]); cw_s = np.array([r[2] for r in rows])
ac_m = np.array([r[3] for r in rows]); ac_s = np.array([r[4] for r in rows])
y = np.arange(len(rows))

tr_m = np.array([r[7] for r in rows]) * 100; tr_s = np.array([r[8] for r in rows]) * 100
va_m = np.array([r[9] for r in rows]) * 100; va_s = np.array([r[10] for r in rows]) * 100
# (D) fine-grained entropy of the hidden maps (Aaronson et al. 2014), layers 2-10, 5 seeds: paper_code/fcn_coffee_rules.py
COF = json.load(open(f"{ROOT}/diagnostics/fcn_coffee_rules.json"))
tag_of = lambda lab: "none" if lab == NONE_LABEL else lab.split()[0].replace("/", "")
en_m = np.array([COF[tag_of(l)]["entropy"][0] for l in labels]); en_s = np.array([COF[tag_of(l)]["entropy"][1] for l in labels])
fig, ((axb, axa), (axd, axc)) = plt.subplots(2, 2, figsize=(16, 12.5), sharey=True)   # A activation, B classifier, C entropy, D accuracy
for ax, m_, s_, none_v, title, xlab in [
    (axa, cw_m, cw_s, none_cw, "(B)  classifier-weight LFPR", "classifier-weight LFPR"),
    (axb, ac_m, ac_s, none_act, "(A)  activation-map LFPR (last hidden layer)", "last-hidden activation LFPR"),
]:
    norm = plt.Normalize(m_.min(), m_.max()); colors = cm.viridis(norm(m_))
    colors[labels.index(NONE_LABEL)] = (0.75, 0.75, 0.75, 1.0)          # the network without lateral inhibition: grey
    ax.barh(y, m_, xerr=s_, color=colors, edgecolor="#333", linewidth=0.6,
            error_kw=dict(lw=1.1, ecolor="#555", capsize=3))
    for i, v in enumerate(m_):
        ax.text(v + s_[i] + 0.008, i, f"{v:.3f}", va="center", fontsize=9.5, weight="bold", color="#222")
    ax.axvline(none_v, color=NONE_C, ls="--", lw=1.8, label=f"plain ({none_v:.3f})")
    ax.set_title(title, fontsize=13, weight="bold", loc="left")
    ax.set_xlabel(xlab + "  (higher = smoother)", fontsize=11)
    ax.set_xlim(0, min(1.0, max(m_.max() + s_.max(), none_v) + 0.09))
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.grid(axis="x", alpha=0.25); ax.legend(loc="lower right", fontsize=10, frameon=False)
h = 0.38
axc.barh(y + h / 2, tr_m, h, xerr=tr_s, color="#bbb", edgecolor="#333", linewidth=0.6, label="training", error_kw=dict(lw=1, capsize=2))
axc.barh(y - h / 2, va_m, h, xerr=va_s, color="#444", edgecolor="#333", linewidth=0.6, label="validation (out of sample)",
         error_kw=dict(lw=1, capsize=2))
XMIN = float(np.floor(min(va_m - va_s) - 1.5))               # axis narrowed around the values
for i in range(len(y)):
    if va_m[i] < XMIN:
        axc.text(XMIN + 0.5, i, f"does not train (training {tr_m[i]:.0f}%, validation {va_m[i]:.0f}%)", va="center", fontsize=8.5, color="#222")
    else:
        axc.text(va_m[i] + va_s[i] + 0.4, i - h / 2, f"{va_m[i]:.1f}", va="center", fontsize=8.5, color="#222")
        axc.text(tr_m[i] + tr_s[i] + 0.4, i + h / 2, f"{tr_m[i]:.1f}", va="center", fontsize=8.5, color="#666")
axc.axvline(np.mean(ntrs) * 100, color=NONE_C, ls=":", lw=1.6, label="plain, training")
axc.axvline(np.mean(nvas) * 100, color=NONE_C, ls="--", lw=1.8, label="plain, validation")
axc.set_title("(D)  accuracy", fontsize=13, weight="bold", loc="left"); axc.set_xlabel("accuracy (%)", fontsize=11)
axc.set_xlim(XMIN, 101.5); axc.grid(axis="x", alpha=0.25)
axc.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=9, frameon=False)
for sp in ("top", "right"): axc.spines[sp].set_visible(False)
norm = plt.Normalize(en_m.min(), en_m.max()); colors = cm.viridis_r(norm(en_m)); colors[labels.index(NONE_LABEL)] = (0.75, 0.75, 0.75, 1.0)
axd.barh(y, en_m, xerr=en_s, color=colors, edgecolor="#333", linewidth=0.6, error_kw=dict(lw=1.1, ecolor="#555", capsize=3))
for i, v in enumerate(en_m):
    axd.text(v + en_s[i] + 0.008, i, f"{v:.2f}", va="center", fontsize=9.5, weight="bold", color="#222")
axd.axvline(COF["none"]["entropy"][0], color=NONE_C, ls="--", lw=1.8, label=f"plain ({COF['none']['entropy'][0]:.2f})")
axd.axvline(COF["input"]["entropy"], color="#2a9d55", ls=":", lw=1.8, label=f"input digit ({COF['input']['entropy']:.2f})")
axd.set_title("(C)  entropy of the hidden maps", fontsize=13, weight="bold", loc="left")
axd.set_xlabel("fine-grained entropy, layers 2-10  (compressed size / noise; lower = simpler)", fontsize=11)
axd.set_xlim(0, 1.0); axd.grid(axis="x", alpha=0.25); axd.legend(loc="lower right", fontsize=10, frameon=False)
for sp in ("top", "right"): axd.spines[sp].set_visible(False)
for a in (axb, axd): a.set_yticks(y); a.set_yticklabels(labels, fontsize=10.5)
for a in (axa, axc): a.tick_params(labelleft=False)
fig.suptitle("MNIST rule sweep: coherence, accuracy and entropy of the hidden maps (10-layer MLP, 300 epochs, 5 seeds)",
             fontsize=13.5, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = f"{ROOT}/paper_figures/out/rule_sweep_300ep.png"
fig.savefig(out, dpi=140, bbox_inches="tight"); print("WROTE", out, flush=True)
