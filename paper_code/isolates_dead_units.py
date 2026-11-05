"""Are the plain FCN's isolated nodes in the co-activation graph of Appendices "grid rule" and "collapse"
(paper_code/er_reviewer_checks.py: top-UNITS-variance units per layer, |corr| edges at <k> = TD) units that have
stopped responding? Same graph and data as er_reviewer_checks.active_map, on the paper FCNs; per dataset and arm:
isolate fraction, fraction of nodes that are dead (activation s.d. over the images < SD_MIN), fraction of isolates
that are dead, and the isolate fraction among live nodes only (graph rebuilt on the live nodes, same <k>).
Writes diagnostics/isolates_dead_units.json.
Run: PYTHONPATH=. python paper_code/isolates_dead_units.py
"""
from conwaynet import REPO_ROOT
import json
import sys

import numpy as np
import torch

sys.path.insert(0, f"{REPO_ROOT}/paper_code")
from conwaynet.config import load_config      # noqa: E402
from conwaynet.data import load_dataset       # noqa: E402
from conwaynet.models import build_model      # noqa: E402

ROOT = REPO_ROOT; DEV = "cuda"; UNITS = 100; TD = 6.0; N = 1000; SD_MIN = 1e-3
DS = [("MNIST", "mnist"), ("KMNIST", "kmnist"), ("FashionMNIST", "fashionmnist"), ("EMNIST", "emnist")]


def acts(m, Xe):
    x = Xe.view(Xe.size(0), -1); out = []
    with torch.no_grad():
        for i, lin in enumerate(m.linears):
            z = torch.tanh(m.norms[i](lin(x)))
            if m.gate is not None:
                z = m.gate(x, z)
            x = z; out.append(z.cpu().numpy())
    return out


def isolates(X):
    C = np.nan_to_num(np.corrcoef(X.T)); np.fill_diagonal(C, 0.)
    n = C.shape[0]; iu = np.triu_indices(n, 1); mg = np.abs(C[iu]); thr = np.sort(mg)[-int(TD * n / 2)]
    deg = np.zeros(n, int); sel = mg >= thr; np.add.at(deg, iu[0][sel], 1); np.add.at(deg, iu[1][sel], 1)
    return deg == 0


def main():
    out = {}
    for dn, ds in DS:
        c0 = load_config(f"{ROOT}/configs/paper/tanh/fcn_{ds}_tanh_ns_ki_ca.json")
        _, _, te, meta = load_dataset(c0["data_path"], c0["batch_size"], c0["data_name"], val_frac=c0["val_frac"],
                                      split_seed=c0["split_seed"], data_norms=c0["data_norms"], output_path=None, num_workers=2)
        Xe = next(iter(te))[0][:N].to(DEV)
        for mech in ("ca", "none"):
            c = load_config(f"{ROOT}/configs/paper/tanh/fcn_{ds}_tanh_ns_ki_{mech}.json")
            m = build_model(c, meta).to(DEV)
            m.load_state_dict(torch.load(f"{ROOT}/paper_runs/fcn_{ds}_tanh_ns_ki_d10/model_{mech}.pth", map_location=DEV)); m.eval()
            A = acts(m, Xe)
            X = np.concatenate([a[:, np.argsort(a.var(0))[-UNITS:]] for a in A], 1)
            dead = X.std(0) < SD_MIN; iso = isolates(X); iso_live = isolates(X[:, ~dead])
            live_per_layer = [int((a.std(0) >= SD_MIN).sum()) for a in A]
            r = dict(iso=float(iso.mean()), dead_nodes=float(dead.mean()), dead_among_iso=float(dead[iso].mean()) if iso.any() else None,
                     iso_live_only=float(iso_live.mean()), live_units_per_layer=live_per_layer)
            out[f"{ds}/{mech}"] = r
            print(f"{dn:12s} {mech:4s}: iso {r['iso']:.2f}  dead nodes {r['dead_nodes']:.2f}  dead among iso {r['dead_among_iso']}  "
                  f"iso (live only) {r['iso_live_only']:.2f}  live units per layer {live_per_layer}", flush=True)
    json.dump(out, open(f"{ROOT}/diagnostics/isolates_dead_units.json", "w"), indent=1)


if __name__ == "__main__":
    main()
