"""The short rule screen of the rule-selection section (text only): the MNIST FCN-10 with each Life-like rule, trained
10 epochs, three seeds per rule (paper_runs/rule_sweep_seeds/<rule>_s<seed>, configs/paper/rule_sweep_seeds/), and the
network without lateral inhibition (NONE_s<seed>). Per rule: classifier-weight LFPR (class mean over the 10 classifier
maps), activation LFPR (class-averaged last hidden map), held-out test accuracy, training accuracy (first 10,000 training
images) and validation accuracy (the run's own split). LFPR = low-frequency power share (DC removed, radius 0.25).
Restored from an earlier figure script (the figure itself is no longer in
the paper); the hidden pass is the FCN's own forward (Linear -> LayerNorm -> tanh -> gate).
Writes diagnostics/rule_screen_10ep.json. Run: PYTHONPATH=. python paper_code/rule_screen_10ep.py
"""
from conwaynet import REPO_ROOT
import numpy as np, torch
from conwaynet.config import load_config
from conwaynet.models import build_model
from conwaynet.data import load_dataset

ROOT = REPO_ROOT; DEV = "cuda" if torch.cuda.is_available() else "cpu"
HW = 28; N = 4000; SWEEP = f"{ROOT}/paper_runs/rule_sweep"
CA = "#d1495b"; NONE_C = "#30638e"

RULES = [  # (config tag, pretty rule string)
    ("B5S3", "B5/S3  (this work)"), ("B3S3", "B3/S3"),
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
            z = torch.tanh(m.norms[j](lin(cur)))
            if m.gate is not None: z = m.gate(cur, z)
            cur = z
        out.append(cur.cpu())
    return torch.cat(out).numpy()


def class_avg_maps(F, Y):
    return np.stack([F[Y == c].mean(0).reshape(HW, HW) for c in range(10)])


# test batch (same MNIST for every rule) — load once via the B5S3 config
c0 = load_config(f"{ROOT}/configs/paper/rule_sweep/fcn_mnist_rule_B5S3_ca.json")
_, _, te, meta = load_dataset(c0["data_path"], c0["batch_size"], c0["data_name"], val_frac=c0["val_frac"],
    split_seed=c0["split_seed"], data_norms=c0["data_norms"], output_path=f"{SWEEP}/B5S3", num_workers=2)
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


SEEDS = (0, 1, 2)  # across-seed aggregation from paper_runs/rule_sweep_seeds/
import json, os


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
        rundir = f"{SWEEP}_seeds/{tag}_s{sd}"
        cfg = f"{ROOT}/configs/paper/rule_sweep_seeds/{tag}_s{sd}_ca.json"
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
    rundir = f"{SWEEP}_seeds/NONE_s{sd}"; cfg = f"{ROOT}/configs/paper/rule_sweep_seeds/none_s{sd}.json"
    if not os.path.exists(f"{rundir}/model_none.pth"):
        continue
    lc, la, ta, va = measure_ckpt(cfg, rundir, "none"); ncws.append(lc); nacts.append(la); naccs.append(acc_of(rundir, "none"))
    ntrs.append(ta); nvas.append(va)
none_cw, none_act = float(np.mean(ncws)), float(np.mean(nacts))
NONE_LABEL = "without lateral inhibition"
rows.append((NONE_LABEL, none_cw, float(np.std(ncws)), none_act, float(np.std(nacts)), float(np.mean(naccs)), float(np.std(naccs)),
             float(np.mean(ntrs)), float(np.std(ntrs)), float(np.mean(nvas)), float(np.std(nvas))))
print(f"{'li=none (no rule)':34s} cw={none_cw:.3f}  act={none_act:.3f}  acc={np.mean(naccs)*100:.1f}", flush=True)

json.dump({r[0]: dict(cw_lfpr=r[1], cw_lfpr_sd=r[2], act_lfpr=r[3], act_lfpr_sd=r[4], test=r[5], test_sd=r[6],
                     train=r[7], train_sd=r[8], val=r[9], val_sd=r[10]) for r in rows},
          open(f"{ROOT}/diagnostics/rule_screen_10ep.json", "w"), indent=1)
print("WROTE diagnostics/rule_screen_10ep.json")
