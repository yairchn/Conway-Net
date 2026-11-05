"""Ensemble coherence of the rule-free blur gates vs the CA gate (paper, 'Coherence across rules' paragraph).

FCN on MNIST, the saved controls of paper_code/fcn_controls.py (paper recipe, seeds 0-2): CA (B5/S3), blur
lambda in {0.5, 0.75, 0.9, 0.95}, and without lateral inhibition. Deep-layer (6-10) unit-wise agreement
between seeds over 1000 test images, trained and at initialisation (rebuilt with the same seed), and the
test accuracy of the network obtained by averaging the three seeds' weights.
Run: PYTHONPATH=. python paper_code/fcn_blur_ensemble.py
"""
from conwaynet import REPO_ROOT
import importlib.util
import itertools
import random

import numpy as np
import torch

from conwaynet.config import load_config
from conwaynet.data import load_dataset

ROOT = REPO_ROOT
spec = importlib.util.spec_from_file_location("fc", f"{ROOT}/paper_code/fcn_controls.py")
fc = importlib.util.module_from_spec(spec); spec.loader.exec_module(fc)
cfg = load_config(fc.CFG.format("ca"))
_, _, te, meta = load_dataset(cfg["data_path"], 500, cfg["data_name"], val_frac=cfg["val_frac"], split_seed=cfg["split_seed"],
                              data_norms=cfg["data_norms"], output_path=None, num_workers=2)
X = torch.cat([x for x, _ in itertools.islice(te, 2)])[:1000].cuda()


@torch.no_grad()
def hidden(m):
    cur = X.view(len(X), -1); out = []
    for i, lin in enumerate(m.linears):
        t = lin(cur); t = m.norms[i](t) if m.norms is not None else t; z = torch.tanh(t)
        if m.gate is not None:
            z = m.gate(cur, z)
        cur = z; out.append(cur.cpu())
    return out


def corr(a, b):
    a = a - a.mean(1, keepdim=True); b = b - b.mean(1, keepdim=True)
    return float(((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-8)).mean())


for arm in ("ca", "blur_0.5", "blur_0.75", "blur_0.9", "blur_0.95", "none"):
    T, U, sds = {}, {}, []
    for s in (0, 1, 2):
        random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
        m, _ = fc.build(arm, s, meta); m.cuda().eval(); U[s] = hidden(m)
        sd = torch.load(f"{fc.OUT}/{arm}_s{s}/model.pth", map_location="cpu"); sds.append(sd)
        m.load_state_dict(sd); m.cuda().eval(); T[s] = hidden(m)
    pr = list(itertools.combinations((0, 1, 2), 2))
    tt = np.mean([[corr(T[a][l], T[b][l]) for l in range(5, 10)] for a, b in pr])
    uu = np.mean([[corr(U[a][l], U[b][l]) for l in range(5, 10)] for a, b in pr])
    avg = {k: torch.stack([d[k].float() for d in sds]).mean(0) for k in sds[0]}
    mw, _ = fc.build(arm, 0, meta); mw.load_state_dict(avg); mw.cuda().eval()
    with torch.no_grad():
        hit = n = 0
        for x, y in te:
            hit += (mw(x.cuda()).argmax(1).cpu() == y).sum().item(); n += len(y)
    print(f"{arm:10s} deep seed agreement trained {tt:.2f} untrained {uu:.2f} | 3-seed weight average test acc {hit / n * 100:.1f}")
